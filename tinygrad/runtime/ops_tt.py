from __future__ import annotations
import os, mmap, struct, time, errno
from dataclasses import replace
from tinygrad.runtime.support.hcq2 import HWQueue, to_name, patch, pack_args, make_program, unwrap_view, get_time_ms
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, KernelInfo, uopfunc
from tinygrad.device import BufferStorage, BufferSpec, Buffer, Device, Allocator, Compiled, host_of
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, getenv, fetch, pluralize, unwrap
from tinygrad.renderer.tt import TTRenderer, MAX_ARGS
from tinygrad.runtime.support.memory import MMIOInterface, TLSFAllocator
from tinygrad.runtime.support.system import FileIOInterface, filter_visible_devices
from tinygrad.engine.realize import get_call_kernel_args

# the firmware is built from https://github.com/boopdotpng/blackhole-py/tree/hcq2-v1/fw, its hw.h has the L1 layout of every core
FW = getenv("TT_FW", "https://github.com/boopdotpng/blackhole-py/releases/download/hcq2-v1")
FW_SHA256 = {"dma_b7": "7892fe757cee462ee5a8823b6fe3e8efb41726830be4795daa45f64ada75dbfa",
             "cp_b7": "b3e964e799a119b914a63b08aaa2dd528e677fc0af403b53fdce8087e3c8812f",
             "worker_b7": "bb6629368299090bf51b2983eace0295434f1f3187368e8656edfdb4f6bbe99e",
             "dma_b8": "5d9ccb066ce7b7403744ec8b7139cffe631916d3e96371efb87f698debfab8d0",
             "cp_b8": "15f85a8f9ceb3e8c1fee218d17edf659cb5fa7238d542663230a709f52a4e781",
             "worker_b8": "59c883e6d72d9e88f0f8753cefee611f557834ba992f8cc4eafb3573d79c51af"}

# tt-kmd, ioctl.h
PIN_PAGES, LOCK_CTL, UNPIN_PAGES, ALLOCATE_TLB, CONFIGURE_TLB, SET_POWER_STATE = 7, 8, 10, 11, 13, 15
TLB_SIZE = 2 << 20
BANKS = {"p100a": 7, "p150a": 8, "p150b": 8} # the dram banks of a card

# a core is (x, y) in translated NoC coordinates, a rect of them (x0, y0, x1, y1)
ARC, TELEMETRY_DATA, TELEMETRY_PTR, TAG_AICLK, TAG_TENSIX_COLS = (8, 0), 0x80030430, 0x80030434, 14, 34
SOFT_RESET, RESET_ALL, RESET_BRISC = 0xffb121b0, 0x47800, 1 << 11
LEADER_COORD, NUM_WORKERS, GRID_COLS, WORK_READY = 0x1c, 0x2c, 0x34, 0x5c # worker L1
WPTR, CP_RECTS, READY, CP_RING, DMA_RING = 0x80, 0x88, 0xa0, 0x5000, 0x9000 # queue core L1, RPTR is the word after WPTR
RING_DEPTH, RING_ENTRY = 64, 8 # in dwords: the address of a cmdbuf, its bytes

PKT_SIZE, PKT_EXEC, PKT_WAIT, PKT_SIGNAL, PKT_TIMESTAMP, PKT_COPY = 64, 1, 3, 4, 5, 7
FLAG_CLEAR_MCAST, MAX_COPY = 1, 1 << 30

# dram is one address space of pages striped over the banks: bit 63, log2 of the page size from bit 56, the offset
DRAM, DRAM_PAGE, DRAM_BANK_SIZE = 1 << 63 | 12 << 56, 4096, 0xff000000
ARENA = 32 << 20 # of pinned host memory

def coord(x:int, y:int) -> int: return y << 6 | x
def pack_rect(x0:int, y0:int, x1:int, y1:int) -> int: return coord(x0, y0) << 12 | coord(x1, y1)
def worker_rects(cols:int) -> list[tuple[int, int, int, int]]: # columns 8 and 9 aren't Tensix, the top two cores of the last are the CP and the DMA
  return [(1, 2, 7, 11), (10, 2, cols + 1, 11), (cols + 2, 2, cols + 2, 9)]

def launch_rects(cols:int, gx:int, gy:int) -> tuple[int, list[tuple[int, int, int, int]]]:
  if gy > 1: # 2D (gx, gy) Tensix grid for multicast matmuls (<= 2 rects, avoiding DRAM cols 8..9 and CP/DMA at (cols+2, 10..11))
    assert gy <= 10 and gx <= (cols if gy <= 8 else cols - 1), f"2D launch ({gx}, {gy}) does not fit {cols}x10 grid"
    rects = [(1, 2, min(gx, 7), 1 + gy)]
    if gx > 7: rects.append((10, 2, gx + 2, 1 + gy))
    return gx * gy, rects
  assert 1 <= gx <= cols * 10 - 2, f"1D launch {gx} exceeds {cols * 10 - 2} workers"
  rects, rem = [], gx
  for x0, y0, x1, y1 in worker_rects(cols):
    if rem == 0: break
    h = y1 - y0 + 1
    if (full:=min(x1 - x0 + 1, rem // h)):
      rects.append((x0, y0, x0 + full - 1, y1))
      rem -= full * h
    if rem and full < x1 - x0 + 1:
      rects.append((x0 + full, y0, x0 + full, y0 + rem - 1))
      rem = 0
  return gx, rects

@uopfunc
def tt_submit(cmdbuf:UOp, addr:UOp, ring:UOp, ptrs:UOp, put:UOp) -> UOp: # ptrs is [wptr, rptr], put the host's copy of wptr
  ring, ptrs = (b.replace(arg=replace(b.arg, volatile=True)) for b in (ring, ptrs)) # mmio: each store is a write to the device
  seq = put.index(0).load() + 1
  # wait for a free ring entry
  loop = UOp.range(UOp(Ops.NOOP).after(start:=get_time_ms(seq)), next(UOp.unique_num), dtype=dtypes.void)
  rptr = ptrs.after(seq, loop).index(1).load()
  free = rptr.backedge(loop, (seq.cast(dtypes.uint32) - rptr > RING_DEPTH) & (get_time_ms(rptr) - start < int(Compiled.wait_timeout_ms)))
  at = (seq % RING_DEPTH).cast(dtypes.int) * RING_ENTRY
  entry = [ring.after(cmdbuf, free).index(at + i).store(w) for i, w in enumerate((addr.cast(dtypes.uint32), (addr >> 32).cast(dtypes.uint32),
                                                                                  UOp.const(cmdbuf.max_numel(), dtypes.uint32)))]
  bell = ptrs.after(*entry).index(0).store(seq.cast(dtypes.uint32))
  return put.after(bell).index(0).store(seq).sink()

class TTQueue(HWQueue): # the CP and the DMA core take the same 64 byte packets, the CP executes and the DMA core copies
  def pkt(self, *words:UOp|int): self.blob += bytes(-self.q(*words) % PKT_SIZE)

  def wait(self, signal:UOp, value:UOp, eq:bool=False): self.pkt(PKT_WAIT, int(eq), signal.getaddr(self.devs), value.cast(dtypes.uint64))
  def signal(self, signal:UOp, value:UOp): self.pkt(PKT_SIGNAL, 0, signal.getaddr(self.devs), value.cast(dtypes.uint64))
  def timestamp(self, signal:UOp): self.pkt(PKT_TIMESTAMP, 0, signal.getaddr(self.devs))

  def copy(self, dst:UOp, src:UOp, sz:int):
    for off in range(0, sz, MAX_COPY):
      self.pkt(PKT_COPY, 0, *(a + UOp.const(off, dtypes.uint64) if off else a for a in (dst.getaddr(self.devs), src.getaddr(self.devs))),
               min(sz - off, MAX_COPY))

  def exec(self, call:UOp, prg:UOp):
    if len(args:=get_call_kernel_args(call, prg, self.devs)) > MAX_ARGS: raise RuntimeError(f"a TT program takes {MAX_ARGS} args, got {len(args)}")
    kernargs = UOp(Ops.LINEAR, src=tuple(pack_args([(8 * i, a) for i, a in enumerate(args)], 8 * MAX_ARGS)), arg="kernargs")
    lib = patch(make_program(prg, len(image:=prg.src[3].arg), self.devs[0]), [], image)
    (gx, gy, gz), _ = prg.arg.launch_dims(call)
    assert gz == 1 and isinstance(gx, int) and isinstance(gy, int), f"unsupported TT launch dims {(gx, gy, gz)}"
    nworkers, rects = launch_rects(Device.target_for(self.devs[0]).get("cols", 14), gx, gy)
    # the CP caches the program in the L1 of the workers by its key
    self.pkt(PKT_EXEC, FLAG_CLEAR_MCAST, lib.getaddr(self.devs), len(image), int.from_bytes(prg.key[:4], "little") or 1,
             kernargs.getaddr(self.devs), nworkers, 8 * MAX_ARGS << 8 | len(rects), *(pack_rect(*r) for r in rects))

  def submit(self, cmdbuf:UOp) -> UOp:
    base, off = unwrap_view(cmdbuf)
    bufs = [UOp.extern((n,), dt, host_of(self.devs[0]), to_name(self.devs[0], name, self.queue))
            for name, n, dt in (("ring", RING_DEPTH * RING_ENTRY, dtypes.uint32), ("ptrs", 2, dtypes.uint32), ("put", 1, dtypes.uint64))]
    return tt_submit(cmdbuf, base.getaddr(self.devs) + off, *bufs)

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("tt_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), lambda s: TTQueue(s).encode()),
  (UPat.sink(arg=KernelInfo("tt_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), lambda s: TTQueue(s).encode()),
])

class TTAllocator(Allocator['TTDevice']):
  def __init__(self, dev:TTDevice):
    super().__init__(dev)
    # a dram buffer is whole stripes of a page per bank
    self.dram = TLSFAllocator(dev.iface.banks * DRAM_BANK_SIZE, block_size=dev.iface.banks * DRAM_PAGE)
    # tt-kmd maps 16 pinned ranges of host memory at most, so the small host buffers share one
    self.arena, self.heap = dev.iface.alloc_host(ARENA), TLSFAllocator(ARENA, block_size=64)

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    # the cpu can't see dram: what it accesses is pinned host memory, which the device reaches over pcie
    if not (options.host or options.cpu_access): return BufferStorage(DRAM | self.dram.alloc(round_up(size, self.dram.block_size)))
    if size > ARENA // 8: return self.dev.iface.alloc_host(size)
    return BufferStorage(self.arena.buf + (off:=self.heap.alloc(round_up(size, 64))), off, unwrap(self.arena.host).view(off, size))

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    if storage.meta is None: self.dram.free(storage.buf ^ DRAM)
    elif isinstance(storage.meta, int): self.heap.free(storage.meta)
    else: self.dev.iface.free_host(*storage.meta)

  def _map(self, buf:Buffer) -> BufferStorage:
    if (host:=buf.get_storage().host) is None or host.addr % mmap.PAGESIZE: raise RuntimeError(f"{buf.device} memory is not page aligned host memory")
    region = (host.addr, round_up(buf.nbytes, mmap.PAGESIZE))
    try: return BufferStorage(self.dev.iface.pin(*region), region)
    except OSError as e:
      if e.errno != errno.ENOSPC: raise
      buf.allocator.free_cache() # the freed buffers it keeps for reuse stay pinned
      return BufferStorage(self.dev.iface.pin(*region), region)
  def _unmap(self, mapping:BufferStorage): self.dev.iface.unpin(*mapping.meta)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class KMDIface:
  def __init__(self, dev, dev_id):
    cards = filter_visible_devices(sorted(int(x) for x in FileIOInterface("/dev/tenstorrent").listdir() if x.isdigit()), "TT")
    if dev_id >= (count:=len(cards)): raise RuntimeError(f"TT:{dev_id} does not exist ({pluralize('device', count)} available)")
    sysfs = f"/sys/class/tenstorrent/tenstorrent!{cards[dev_id]}"
    if not FileIOInterface.exists(iommu:=f"{sysfs}/device/iommu_group/type") or not FileIOInterface(iommu).read().startswith("DMA"):
      raise RuntimeError("TT pins host memory through the IOMMU, enable it")
    if (card:=FileIOInterface(f"{sysfs}/tt_card_type").read().strip()) not in BANKS: raise RuntimeError(f"unsupported TT card {card}")
    self.dev, self.count, self.card, self.banks = dev, count, card, BANKS[card]

    self.fd = FileIOInterface(f"/dev/tenstorrent/{cards[dev_id]}", os.O_RDWR | os.O_CLOEXEC | os.O_APPEND) # O_APPEND: we set the power state
    if not self.ioctl(LOCK_CTL, "IIB3xB3x", 4, 0, 20, 0)[-1]: raise RuntimeError(f"TT:{dev_id} is in use by another process") # chip in use
    self.set_power(True)
    (self.tlb, self.win), self.target = self.window(), None
    self.cols = (self.telemetry(TAG_TENSIX_COLS) & 0x3fff).bit_count()
    self.boot()

  def ioctl(self, nr:int, fmt:str, *args:int) -> tuple[int, ...]:
    self.fd.ioctl(0xfa00 | nr, data:=bytearray(struct.pack("<" + fmt, *args)))
    return struct.unpack("<" + fmt, data)
  def set_power(self, on:bool): self.ioctl(SET_POWER_STATE, "IIBBH28x", 40, 0, 0, 15, 7 if on else 0)
  def pin(self, va:int, size:int) -> int: return self.ioctl(PIN_PAGES, "IIQQQQ", 16, 2, va, size, 0, 0)[-1] # as noc dma, the noc address
  def unpin(self, va:int, size:int): self.ioctl(UNPIN_PAGES, "QQQ", va, size, 0)

  def alloc_host(self, size:int) -> BufferStorage:
    va = FileIOInterface.anon_mmap(0, size:=round_up(size, mmap.PAGESIZE), mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, 0)
    return BufferStorage(self.pin(va, size), (va, size), MMIOInterface(va, size))
  def free_host(self, va:int, size:int):
    self.unpin(va, size)
    FileIOInterface.munmap(va, size)

  # a tlb window is 2MB of a core's address space, or written to every core of a rect

  def window(self, core:tuple[int, ...]|None=None) -> tuple[int, MMIOInterface]:
    _, _, tlb, _, offset, _, _ = self.ioctl(ALLOCATE_TLB, "QQIIQQQ", TLB_SIZE, 0, 0, 0, 0, 0, 0)
    if core is not None: self.configure(tlb, core, 0)
    return tlb, MMIOInterface(self.fd.mmap(0, TLB_SIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset), TLB_SIZE)
  def configure(self, tlb:int, core:tuple[int, ...], addr:int):
    x0, y0, x1, y1 = core * (4 // len(core))
    self.ioctl(CONFIGURE_TLB, "IIQ4HBBB21x", tlb, 0, addr, x1, y1, x0, y0, 0, len(core) == 4, 1)

  def at(self, core:tuple[int, ...], addr:int, fmt='B') -> MMIOInterface: # the scratch window, from addr
    if self.target != (target:=(core, addr & -TLB_SIZE)): self.configure(self.tlb, *target)
    self.target = target
    return self.win.view(addr % TLB_SIZE, fmt=fmt)
  def read32(self, core:tuple[int, ...], addr:int) -> int: return self.at(core, addr, 'I')[0]
  def write32(self, core:tuple[int, ...], addr:int, val:int): self.at(core, addr, 'I')[0] = val

  def telemetry(self, tag:int) -> int:
    table, data = self.read32(ARC, TELEMETRY_PTR), self.read32(ARC, TELEMETRY_DATA)
    for entry in (self.read32(ARC, table + 8 + 4 * i) for i in range(self.read32(ARC, table + 4))):
      if entry & 0xffff == tag: return self.read32(ARC, data + 4 * (entry >> 16))
    raise RuntimeError(f"the ARC has no telemetry tag {tag}")

  def boot(self):
    self.cp, self.dma, rects = (self.cols + 2, 10), (self.cols + 2, 11), worker_rects(self.cols)
    workers = [(x, y) for x0, y0, x1, y1 in rects for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    dma, cp, worker = (bytearray(fetch(f"{FW}/{n}.bin", subdir="fw", sha256=FW_SHA256[n]).read_bytes())
                       for n in (f"{name}_b{self.banks}" for name in ("dma", "cp", "worker")))
    struct.pack_into(f"<{1 + len(rects)}I", cp, CP_RECTS, len(rects), *(pack_rect(*r) for r in rects))
    struct.pack_into("<I", worker, LEADER_COORD, coord(*self.cp))
    struct.pack_into("<I", worker, NUM_WORKERS, len(workers))
    struct.pack_into("<I", worker, GRID_COLS, self.cols)

    # an image is the L1 of a core from 0, where the jump to its firmware is
    self.reset()
    for core, image in ((self.dma, dma), (self.cp, cp), *((r, worker) for r in rects)): self.at(core, 0)[:len(image)] = image
    for core in (self.dma, self.cp, *rects): self.write32(core, SOFT_RESET, RESET_ALL & ~RESET_BRISC)
    st = time.perf_counter()
    for core, addr in ((self.dma, READY), (self.cp, READY), *((w, WORK_READY) for w in workers)):
      while self.read32(core, addr) != 1:
        if time.perf_counter() - st > 5: raise RuntimeError(f"TT core {core} did not boot")

  def reset(self): self.write32((1, 2, self.cols + 2, 11), SOFT_RESET, RESET_ALL | 1 << 15 | 1 << 16) # every risc held, the Tensix sources too

  def device_fini(self):
    self.reset()
    self.set_power(False)

class TTDevice(Compiled):
  renderers = [TTRenderer]
  ifaces = [KMDIface]

  def __init__(self, device:str=""):
    self.iface = self._select_iface(device)
    super().__init__(device, TTAllocator(self), None, arch=self.iface.card, props={"cols": self.iface.cols, "banks": self.iface.banks})
    self.timestamp_divider = float(self.iface.telemetry(TAG_AICLK)) # the wall clock ticks at aiclk, in MHz

    # a queue is a ring and its pointers in the L1 of the core that runs it, with the host's copy of the write pointer
    for queue, core, ring in (("COMPUTE:0", self.iface.cp, CP_RING), ("COPY:0", self.iface.dma, DMA_RING)):
      l1 = self.iface.window(core)[1]
      Compiled.link_bufs |= {to_name(device, name, queue): Buffer(self.host, n, options=BufferSpec(external_ptr=l1.addr + off), preallocate=True)
                             for name, off, n in (("ring", ring, RING_DEPTH * RING_ENTRY * 4), ("ptrs", WPTR, 8))}
      Compiled.link_bufs |= {to_name(device, "put", queue): Buffer(self.host, 8, initial_value=bytes(8))}
