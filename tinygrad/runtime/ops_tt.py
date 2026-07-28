from __future__ import annotations
import array, functools, glob
from dataclasses import dataclass
from typing import Literal

from tinygrad.device import BufferSpec, TinyELF
from tinygrad.helpers import round_up
from tinygrad.renderer.tensix import TensixRenderer
from tinygrad.runtime.support.hcq import HCQAllocator, HCQArgsState, HCQBuffer, HCQCompiled, HCQProgram, HCQSignal, HWQueue
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.support.tt.chip import TTChip
from tinygrad.runtime.support.tt.cq import MAX_WRITE_SIZE, CommandQueue, DramCopy, McastWrite, Run, Signal, UnicastWrite, _rectangles
from tinygrad.runtime.support.tt.firmware.consts import Firmware, KERNEL_ROLES, TensixL1
from tinygrad.runtime.support.tt.isa import R, RV32
from tinygrad.runtime.support.tt.pcie import PCIDevice
from tinygrad.runtime.support.tt.program import decode_tt_program

DRAM_PAGE_SIZE = 4096
DRAM_BANKS = 7
DRAM_START, DRAM_END = 0x40, 1 << 32
TT_VA_START, TT_VA_SIZE = 0x1000_0000_0000, 1 << 44


@dataclass(frozen=True)
class TTAllocation:
  kind:Literal["dram", "host"]
  va_addr:int
  size:int                 # exact tinygrad-visible bytes
  capacity:int             # private physical slack, never tensor-visible
  dram_addr:int|None = None
  sysmem_offset:int|None = None
  page_size:int = DRAM_PAGE_SIZE
  banks:int = DRAM_BANKS


class TTAllocator(HCQAllocator['TTDevice']):
  """Exact-size HCQ buffers backed by interleaved Blackhole DRAM or pinned sysmem."""

  def __init__(self, dev:TTDevice, batch_size:int=2 << 20, batch_cnt:int=32):
    self.sysmem = dev.iface.pcie.sysmem
    if self.sysmem is None: raise RuntimeError("TT sysmem is closed")

    host_base = round_up(self.sysmem.allocator.ptr, self.sysmem.PAGE_SIZE)
    host_size = self.sysmem.size - host_base
    if host_size <= 0: raise MemoryError("TT pinned sysmem has no HCQ allocation arena")
    # CommandQueue has made all fixed allocations. TTAllocator exclusively owns the remainder.
    self.sysmem.allocator.ptr = self.sysmem.size
    self.host_mm = TLSFAllocator(host_size, base=host_base, block_size=16)
    self.dram_mm = TLSFAllocator(DRAM_END - DRAM_START, base=DRAM_START, block_size=64)
    self.va_mm = TLSFAllocator(TT_VA_SIZE, base=TT_VA_START, block_size=64)
    super().__init__(dev, batch_size=batch_size, batch_cnt=batch_cnt, supports_transfer=False, supports_copy_from_disk=False)

  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: raise NotImplementedError("TT external pointers are not supported")
    if options.host or options.cpu_access:
      capacity = round_up(size, 16)
      offset = self.host_mm.alloc(capacity, align=64)
      meta = TTAllocation("host", self.sysmem.noc_addr + offset, size, capacity, sysmem_offset=offset)
      return HCQBuffer(meta.va_addr, size, meta=meta, view=self.sysmem.view.view(offset, size), owner=self.dev)

    # Every bank reserves the same local span. Logical pages rotate over banks;
    # the final page is short and the unused bytes are private allocation slack.
    page_count = (size + DRAM_PAGE_SIZE - 1) // DRAM_PAGE_SIZE
    bank_capacity = round_up((page_count + DRAM_BANKS - 1) // DRAM_BANKS * DRAM_PAGE_SIZE, 64)
    dram_addr = self.dram_mm.alloc(bank_capacity, align=64)
    try: va_addr = self.va_mm.alloc(size, align=64)
    except Exception:
      self.dram_mm.free(dram_addr)
      raise
    meta = TTAllocation("dram", va_addr, size, bank_capacity, dram_addr=dram_addr)
    return HCQBuffer(va_addr, size, meta=meta, owner=self.dev)

  def _do_free(self, buf:HCQBuffer, options:BufferSpec):
    meta:TTAllocation = buf.meta
    if meta.kind == "host":
      assert meta.sysmem_offset is not None
      self.host_mm.free(meta.sysmem_offset)
    else:
      assert meta.dram_addr is not None
      self.dram_mm.free(meta.dram_addr)
      self.va_mm.free(meta.va_addr)

  @staticmethod
  def _root_info(buf:HCQBuffer) -> tuple[TTAllocation, int]:
    meta = buf.base.meta
    if not isinstance(meta, TTAllocation): raise TypeError("buffer is not a TT allocation")
    offset = int(buf.va_addr) - meta.va_addr
    if offset < 0 or offset + buf.size > meta.size: raise ValueError("TT buffer view is outside its allocation")
    return meta, offset

  @classmethod
  def copy_commands(cls, dest:HCQBuffer, src:HCQBuffer, size:int) -> list[DramCopy]:
    dmeta, doff = cls._root_info(dest)
    smeta, soff = cls._root_info(src)
    if size < 0 or size > dest.size or size > src.size: raise ValueError("TT copy exceeds a buffer view")
    if {dmeta.kind, smeta.kind} != {"dram", "host"}: raise NotImplementedError("TT copies currently require one host and one DRAM buffer")

    dram_meta, dram_off, host_meta, host_off, direction = \
      (dmeta, doff, smeta, soff, 0) if dmeta.kind == "dram" else (smeta, soff, dmeta, doff, 1)
    if (dram_off | host_off) & 15: raise ValueError("TT host/DRAM copies require 16-byte-aligned offsets")
    if size & 15 and dram_off + size != dram_meta.size:
      raise ValueError("a short TT transfer is valid only at the end of a DRAM allocation")
    assert dram_meta.dram_addr is not None

    commands:list[DramCopy] = []
    copied = 0
    while copied < size:
      logical = dram_off + copied
      page, within = divmod(logical, dram_meta.page_size)
      valid = min(size - copied, dram_meta.page_size - within)
      transfer = round_up(valid, 16)
      bank = page % dram_meta.banks
      physical = dram_meta.dram_addr + (page // dram_meta.banks) * dram_meta.page_size + within
      host_addr = host_meta.va_addr + host_off + copied
      if host_off + copied + transfer > host_meta.capacity: raise ValueError("TT short transfer exceeds host allocation slack")
      commands.append(DramCopy(physical, host_addr, transfer, 1, 1, direction, bank))
      copied += valid
    return commands


class TTSignal(HCQSignal['TTDevice']):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, "timestamp_divider": 1350})


class TTArgsState(HCQArgsState['TTProgram']):
  """Initial direct-launch ABI: one packed 32-bit DRAM address per buffer, then scalar words."""
  def __init__(self, buf:HCQBuffer, prg:TTProgram, bufs:tuple[HCQBuffer, ...], vals=()):
    super().__init__(buf, prg, bufs, vals)
    words = []
    for arg in bufs:
      meta, offset = TTAllocator._root_info(arg)
      if meta.kind != "dram" or meta.dram_addr is None: raise NotImplementedError("TT kernels currently accept only DRAM buffer arguments")
      page, within = divmod(offset, meta.page_size)
      physical = meta.dram_addr + (page // meta.banks) * meta.page_size + within
      bank = page % meta.banks
      if physical & 7: raise ValueError("TT direct-launch buffer views must preserve 8-byte address alignment")
      words.append(physical | bank)
    if any(not isinstance(value, int) or not 0 <= value < 1 << 32 for value in vals):
      raise NotImplementedError("TT direct-launch scalar arguments must be concrete 32-bit values")
    words.extend(vals)
    if len(words) > TensixL1.PARAM_SLOTS: raise ValueError(f"TT direct launch supports at most {TensixL1.PARAM_SLOTS} argument words")
    self.word_count = len(words)
    if words: self.buf.cpu_view().view(size=len(words) * 4, fmt="I")[:] = array.array("I", words)


class TTProgram(HCQProgram['TTDevice']):
  def __init__(self, dev:TTDevice, obj:TinyELF):
    self.images = decode_tt_program(obj.lib)
    for role, image in self.images.items():
      if len(image) > TensixL1.WORKER_TEXT_SIZE[role]: raise ValueError(f"TT {role} image exceeds its direct-launch L1 partition")
    super().__init__(TTArgsState, dev, obj.name, TensixL1.PARAM_SIZE, lib=obj.lib)


_RETURN_KERNEL = {role: RV32().jal(R.ZERO, Firmware.TEXT[role][0] - TensixL1.WORKER_TEXT_BASE[role]).to_bytes(4, "little")
                  for role in KERNEL_ROLES}


class TTCopyQueue(HWQueue):
  def __init__(self, dev:TTDevice): self.dev = dev; super().__init__()

  def wait(self, signal:TTSignal, value):
    if signal.owner not in (None, self.dev): raise NotImplementedError("cross-device TT waits are not implemented")
    # All current TT work shares one ordered resident command queue.
    return self

  def copy(self, dest:HCQBuffer, src:HCQBuffer, copy_size:int):
    self._q.extend(self.dev.allocator.copy_commands(dest, src, copy_size))
    return self

  def timestamp(self, signal:TTSignal):
    self._q.append(Signal(int(signal.value_addr), signal.value))
    return self

  def signal(self, signal:TTSignal, value=0):
    if not isinstance(value, int): raise NotImplementedError("symbolic TT signal values are not implemented")
    self._q.append(Signal(int(signal.value_addr), value))
    return self

  def _submit(self, dev:TTDevice):
    if dev is not self.dev: raise ValueError("TT queue submitted to the wrong device")
    dev.iface.cq.enqueue(tuple(self._q), completion=False)


class TTComputeQueue(TTCopyQueue):
  def memory_barrier(self): return self  # one ordered TT issue stream currently provides the required ordering

  def exec(self, prg:TTProgram, args_state:TTArgsState, global_size, local_size):
    if any(not isinstance(value, int) for value in (*global_size, *local_size)):
      raise NotImplementedError("symbolic TT launch geometry is not implemented")
    if tuple(local_size) != (1, 1, 1): raise ValueError("initial TT launches require local_size=(1, 1, 1)")
    core_count = global_size[0] * global_size[1] * global_size[2]
    if not 0 < core_count <= len(self.dev.iface.pcie.cores): raise ValueError("TT launch core count is outside the usable worker set")
    cores = tuple(self.dev.iface.pcie.cores[:core_count])

    for role in KERNEL_ROLES:
      image = prg.images[role] or _RETURN_KERNEL[role]
      for offset in range(0, len(image), MAX_WRITE_SIZE):
        chunk = image[offset:offset+MAX_WRITE_SIZE]
        address = TensixL1.WORKER_TEXT_BASE[role] + offset
        self._q.append(UnicastWrite(cores, address, (chunk,) * len(cores)) if len(cores) == 1 else
                       McastWrite(_rectangles(cores), address, chunk))
    if args_state.word_count:
      table = bytes(args_state.buf.cpu_view().view(size=args_state.word_count * 4, fmt="B")[:])
      self._q.append(UnicastWrite(cores, TensixL1.PARAM_BASE, (table,) * len(cores)))
    self._q.append(Run(cores))
    return self


class KMDIface:
  """The sole TT interface: tt-kmd transport plus userspace chip/CQ boot."""
  def __init__(self, dev:TTDevice, device_id:int):
    self.dev, self.count = dev, len(glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
    self.pcie = PCIDevice(device_id)
    self.cq = self.chip = None
    try:
      self.cq = CommandQueue(self.pcie)
      self.chip = TTChip(self.pcie)
      self.chip.boot(self.cq)
    except Exception:
      self.device_fini()
      raise

  def device_fini(self):
    if self.chip is not None:
      self.chip.fini()
      self.chip = None
    if self.cq is not None:
      self.cq.close()
      self.cq = None
    self.pcie.close()


class TTDevice(HCQCompiled[TTSignal]):
  """tinygrad registration point for the Tenstorrent backend.

  Allocation, host copies, and direct five-image worker launches use the
  resident TT command queue. Generic Tensix rendering/codegen lands separately.
  """
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = KMDIface(self, self.device_id)
    super().__init__(device, TTAllocator(self), [TensixRenderer], TTProgram, TTSignal, functools.partial(TTComputeQueue, self),
                     functools.partial(TTCopyQueue, self), arch="blackhole")
