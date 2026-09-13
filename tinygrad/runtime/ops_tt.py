from __future__ import annotations
import functools, json, struct
from typing import Any
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.runtime.support.hcq2 import HWQueue, encode_submit, patch, STAGING_SIZE, unwrap_view
from tinygrad.runtime.support.memory import TLSFAllocator, MMIOInterface
from tinygrad.runtime.support.tt.device import TTInterface, jal
from tinygrad.runtime.support.tt.consts import Firmware, TensixL1, KERNEL_ROLES

# DRAM is a byte-addressed logical space striped in 2 KiB pages across all banks.
# Sysmem retains the driver's NoC encoding. L1 uses (x | y<<6)<<32 | offset.
DRAM = 1 << 63
RING_SIZE = 4 << 20

def tt_program(images:dict[tuple[int,int], dict[str,bytes]], params:dict[tuple[int,int], tuple[int,...]]|None=None) -> bytes:
  """Package worker code. Runtime pointers (u64) and scalars follow each core's u32 constants."""
  if not images: raise ValueError('a program needs at least one worker')
  return b'TTP1' + json.dumps([(c, {r:b.hex() for r,b in roles.items()}, (params or {}).get(c, ())) for c,roles in images.items()],
                              separators=(',', ':')).encode()

@functools.cache
def program_records(lib:bytes):
  if lib[:4] != b'TTP1': raise ValueError('TT requires a TTP1 worker bundle; tensor code generation is not implemented')
  records = []
  for core, roles, params in json.loads(lib[4:]):
    if set(roles) - set(KERNEL_ROLES): raise ValueError('unknown worker role')
    images = {r:bytes.fromhex(roles[r]) if r in roles else jal(Firmware.TEXT[r][0] - TensixL1.WORKER_TEXT_BASE[r]) for r in KERNEL_ROLES}
    for r, image in images.items():
      if not image or len(image)%4 or len(image)>TensixL1.WORKER_TEXT_SIZE[r]: raise ValueError('invalid worker image size')
    if any(type(v) is not int or not 0<=v<1<<32 for v in params): raise ValueError('constants must be u32')
    records.append((tuple(core), images, tuple(params)))
  if not records or len({c for c,_,_ in records}) != len(records): raise ValueError('invalid worker list')
  return records

class TTQueue(HWQueue):
  dev:TTDevice

  def packet(self, op:int, *words:Any):
    start = len(self.blob)
    self.q(op, 64, *words)
    if len(self.blob)-start > 64: raise ValueError('packet exceeds 64 bytes')
    self.q(UOp(Ops.BINARY, arg=bytes(64-(len(self.blob)-start))))

  def wait(self, signal:UOp, value:UOp, eq:bool=False):
    self.packet(8, signal.getaddr(self.devs), value.cast(dtypes.uint64), int(eq))
  def signal(self, signal:UOp, value:UOp): self.packet(5, signal.getaddr(self.devs), value.cast(dtypes.uint64))
  def timestamp(self, signal:UOp): self.packet(9, signal.getaddr(self.devs)+8)
  def memory_barrier(self): pass # dispatch drains both DMA engines before releasing workers or evaluating a wait

  def copy(self, call:UOp):
    dest, src = get_call_arg_uops(call)
    if (size:=dest.nbytes()) >= 1<<32: raise ValueError('split DMA transfers larger than 4 GiB')
    self.packet(10, 0, 0, src.getaddr(self.devs), dest.getaddr(self.devs), size)

  def write(self, core:tuple[int,int], addr:int, words:list[Any]):
    payload = sum(w.nbytes() if isinstance(w,UOp) and w.op is Ops.BINARY else w.dtype.itemsize if isinstance(w,UOp) else 4 for w in words)
    size = 64+round_up(payload,64)
    self.q(1 | 1<<16, size, addr, payload, core[0] | core[1]<<6, UOp(Ops.BINARY,arg=bytes(44)), *words,
           UOp(Ops.BINARY,arg=bytes(-payload%64)))

  def exec(self, call:UOp, prg:UOp):
    records = program_records(prg.src[3].arg)
    args = [get_call_arg_uops(call)[i].getaddr(self.devs) for i in prg.arg.globals]
    vals = [v.cast(dtypes.uint32) for v in get_call_var_uops(call, prg)]
    for core, images, constants in records:
      if core not in self.dev.iface.pcie.cores: raise ValueError(f'unavailable worker {core}')
      for role, image in images.items():
        for off in range(0,len(image),16384):
          self.write(core, TensixL1.WORKER_TEXT_BASE[role]+off, [UOp(Ops.BINARY,arg=image[off:off+16384])])
      words = [*constants, *args, *vals]
      n = sum(w.dtype.itemsize if isinstance(w,UOp) else 4 for w in words)
      if n > TensixL1.PARAM_SIZE: raise ValueError('worker arguments exceed parameter table')
      self.write(core, TensixL1.PARAM_BASE, words + [0]*((TensixL1.PARAM_SIZE-n)//4))
    n = len(records)
    size = round_up(24+8*n,64)
    self.q(3 | n<<16, size, 0, n, 0, 0)
    for (x,y),_,_ in records: self.q(x|y<<6, x|y<<6)
    self.q(UOp(Ops.BINARY,arg=bytes(size-24-8*n)))

  def submit(self, cmdbuf:UOp) -> UOp:
    ring, read, doorbell, put = [UOp.placeholder((size,),dt,device=self.devs,volatile=True,tag='tt_'+name)
      for name,size,dt in (('ring',RING_SIZE//4,dtypes.uint32),('read',1,dtypes.uint64),('doorbell',1,dtypes.uint64),('put',1,dtypes.uint64))]
    p = put.index(0).load()
    # Each ring entry is one 64-byte indirect descriptor, so none straddle wrap.
    done = read.after(cmdbuf, loop:=UOp.loop(10)).index(0).load()
    ready = done.end(loop, p+64-done > RING_SIZE)
    desc = UOp.placeholder((64,),dtypes.uint8,device=self.devs,tag='tt_ib')
    base, off = unwrap_view(cmdbuf)
    desc = patch(desc, [(8,base.getaddr(self.devs)+off)], struct.pack('<4I',6,64,0,0)+struct.pack('<I',cmdbuf.nbytes())+bytes(44))
    i = UOp.range(16,11,src=(ready,))
    written = ring.after(ready).index((p % RING_SIZE).cast(dtypes.int)//4+i).store(desc.bitcast(dtypes.uint32).index(i).load()).end(i)
    return doorbell.after(put.after(written.barrier()).index(0).store(p+64)).index(0).store(p+64)

class TTAllocator(Allocator['TTDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if (addr:=options.external_ptr) is not None:
      mem = self.dev.iface.sysmem
      if mem.noc_addr <= addr and addr+size <= mem.noc_addr+mem.size:
        return BufferStorage(addr, host=MMIOInterface(mem.addr+addr-mem.noc_addr,size))
      if addr & DRAM:
        if addr+size > DRAM+len(self.dev.iface.pcie.dram_endpoints)*(1<<32): raise ValueError('DRAM view out of bounds')
      elif (addr>>32 & 63, addr>>38) not in self.dev.iface.pcie.cores or (addr & 0xffffffff)+size > TensixL1.SIZE:
        raise ValueError('invalid L1 view')
      return BufferStorage(addr)
    if options.host or options.cpu_access:
      off = self.dev.sysmem.alloc(round_up(size,64),256)
      return BufferStorage(self.dev.iface.sysmem.noc_addr+off, ('host',off),
                           MMIOInterface(self.dev.iface.sysmem.addr+off,size))
    stripe = 2048*len(self.dev.iface.pcie.dram_endpoints)
    off = self.dev.dram.alloc(round_up(size,stripe),stripe)
    return BufferStorage(DRAM+off, ('dram',off))

  def _free(self, storage:BufferStorage, options:BufferSpec):
    if self.dev.iface.pcie.fd < 0 or storage.meta is None: return
    self.dev.synchronize()
    kind, off = storage.meta
    (self.dev.sysmem if kind=='host' else self.dev.dram).free(off)

  def _map(self, buf:Buffer) -> BufferStorage:
    sysmem = self.dev.iface.sysmem
    if buf.get_storage().host is not None and sysmem.addr <= buf.host.addr and buf.host.addr+buf.nbytes <= sysmem.addr+sysmem.size:
      return BufferStorage(sysmem.noc_addr+buf.host.addr-sysmem.addr)
    raise RuntimeError('TT DMA requires the pinned sysmem staging pool')

  def _offset(self, buf:int, size:int, offset:int) -> int: return buf+offset

class TTDevice(Compiled):
  timestamp_divider = 1350.0
  rtalloc_size = 16<<20
  copy_queue = 'COMPUTE:0' # one FIFO fronts the compute dispatcher and two asynchronous DMA engines
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION,arg='submit_tt_compute',name='submit'),lambda ctx,submit: encode_submit(TTQueue(ctx,submit))),
  ])

  def __init__(self, device:str):
    self.iface = TTInterface(int(device.split(':')[1]) if ':' in device else 0)
    self.iface.peer_group = device # separate cards use host staging, not peer NoC addresses
    self.sysmem = TLSFAllocator(self.iface.sysmem.size, block_size=64)
    stripe = 2048*len(self.iface.pcie.dram_endpoints)
    self.dram = TLSFAllocator(len(self.iface.pcie.dram_endpoints)*(1<<32)-stripe,base=stripe,block_size=2048)
    super().__init__(device, TTAllocator(self), [], None, arch='blackhole')
    def hostbuf(size, dtype):
      spec = BufferSpec(host=True)
      buf = Buffer(device,size//dtype.itemsize,dtype,options=spec,opaque=self.allocator._alloc(size,spec))
      buf.host[:] = memoryview(bytes(size))
      return buf
    self.ring, self.read, self.put = hostbuf(RING_SIZE,dtypes.uint32), hostbuf(8,dtypes.uint64), hostbuf(8,dtypes.uint64)
    self.doorbell = Buffer(device,1,dtypes.uint64,opaque=BufferStorage(0,host=MMIOInterface(self.iface.prefetch.addr+0x1000,8)))
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM,name='b'),lambda ctx,b: getattr(ctx,b.tag[3:]) if b.tag in ('tt_ring','tt_read','tt_put','tt_doorbell') else None),
    ]) + self.pm_bufferize
    try: self.iface.boot(self.ring._buf,self.read._buf)
    except Exception:
      self.iface.device_fini()
      raise

  @functools.cache
  def staging_buffer(self) -> Buffer:
    return Buffer(self.device,STAGING_SIZE,dtypes.uint8,options=BufferSpec(host=True),preallocate=True)

  def synchronize(self, timeout:int|None=None):
    if self.iface.pcie.fd >= 0: super().synchronize(timeout)

  def finalize(self):
    self.synchronize()
    for d in Device._opened_devices: Device[d].pending.pop(self, None)
    self.iface.device_fini()

  def _at_profile_finalize(self): pass # no Tensor renderer yet; queue timestamps use Compiled.collect_prof
