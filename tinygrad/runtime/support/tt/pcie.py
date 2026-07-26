from __future__ import annotations
import builtins, ctypes, mmap, os

from tinygrad.helpers import suppress_finalizing
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator

IOCTL_MAGIC = 0xFA

# Virtual NoC 0/1 coordinates for one usable endpoint in each DRAM bank.
P100_DRAM_ENDPOINTS = (
  ((18, 14), (18, 13)), ((18, 15), (18, 16)), ((18, 18), (18, 19)),
  ((17, 21), (17, 22)), ((17, 14), (17, 13)), ((17, 17), (17, 16)),
  ((17, 20), (17, 19)),
)
P100_WORKER_CORES = tuple(
  (x, y) for x in (*range(1, 8), *range(10, 15)) for y in range(2, 12)
  if (x, y) not in ((14, 2), (14, 3), (14, 4))
)

def _TT_IOCTL(nr, payload_type, result=None, **defaults):
  def call(fd:FileIOInterface, **kwargs):
    payload = payload_type(**(defaults | kwargs))
    fd.ioctl((IOCTL_MAGIC << 8) | nr, payload)
    return getattr(payload, result) if result else None
  return call

class PinPagesIn(ctypes.Structure):
  _fields_ = [
    ("_output_size_bytes", ctypes.c_uint32),
    ("_flags", ctypes.c_uint32),
    ("virtual_address", ctypes.c_uint64),
    ("size", ctypes.c_uint64),
  ]

class PinPagesOut(ctypes.Structure):
  _fields_ = [("_physical_address", ctypes.c_uint64), ("noc_address", ctypes.c_uint64)]

class PinPagesPayload(ctypes.Structure):
  _anonymous_ = ("in_", "out")
  _fields_ = [("in_", PinPagesIn), ("out", PinPagesOut)]

class UnpinPagesIn(ctypes.Structure):
  _fields_ = [("virtual_address", ctypes.c_uint64), ("size", ctypes.c_uint64), ("_reserved", ctypes.c_uint64)]

class UnpinPagesPayload(ctypes.Structure):
  _anonymous_ = ("in_",)
  _fields_ = [("in_", UnpinPagesIn)]

class AllocateTlbIn(ctypes.Structure):
  _fields_ = [("_size", ctypes.c_uint64), ("_reserved", ctypes.c_uint64)]

class AllocateTlbOut(ctypes.Structure):
  _fields_ = [
    ("id", ctypes.c_uint32),
    ("_reserved0", ctypes.c_uint32),
    ("mmap_offset_uc", ctypes.c_uint64),
    ("_mmap_offset_wc", ctypes.c_uint64),
    ("_reserved1", ctypes.c_uint64),
  ]

class AllocateTlbPayload(ctypes.Structure):
  _anonymous_ = ("in_", "out")
  _fields_ = [("in_", AllocateTlbIn), ("out", AllocateTlbOut)]

class FreeTlbIn(ctypes.Structure):
  _fields_ = [("id", ctypes.c_uint32)]

class FreeTlbPayload(ctypes.Structure):
  _anonymous_ = ("in_",)
  _fields_ = [("in_", FreeTlbIn)]

class NocTlbConfig(ctypes.Structure):
  _fields_ = [
    ("addr", ctypes.c_uint64),
    ("x_end", ctypes.c_uint16),
    ("y_end", ctypes.c_uint16),
    ("x_start", ctypes.c_uint16),
    ("y_start", ctypes.c_uint16),
    ("_noc_mcast", ctypes.c_uint8 * 2),
    ("_ordering", ctypes.c_uint8),
    ("_unused", ctypes.c_uint8 * 5),
    ("_reserved", ctypes.c_uint32 * 2),
  ]

class ConfigureTlbIn(ctypes.Structure):
  _anonymous_ = ("config",)
  _fields_ = [("id", ctypes.c_uint32), ("_reserved", ctypes.c_uint32), ("config", NocTlbConfig)]

class ConfigureTlbPayload(ctypes.Structure):
  _anonymous_ = ("in_",)
  _fields_ = [("in_", ConfigureTlbIn), ("_out_reserved", ctypes.c_uint64)]

  def __init__(self, id, addr, start, end=None):  # noqa: A002
    end = start if end is None else end
    super().__init__(in_=ConfigureTlbIn(id=id, config=NocTlbConfig(
      addr=addr, x_end=end[0], y_end=end[1], x_start=start[0], y_start=start[1],
      _noc_mcast=(ctypes.c_uint8 * 2)(0, start != end), _ordering=1)))

class PowerState(ctypes.Structure):
  _fields_ = [
    ("_argsz", ctypes.c_uint32),
    ("_unused", ctypes.c_uint8 * 5),
    ("_validity", ctypes.c_uint8),
    ("power_flags", ctypes.c_uint16),
    ("_power_settings", ctypes.c_uint16 * 14),
  ]

PinPages = _TT_IOCTL(7, PinPagesPayload, "out", _output_size_bytes=ctypes.sizeof(PinPagesOut), _flags=2)
UnpinPages = _TT_IOCTL(10, UnpinPagesPayload)
AllocateTlb = _TT_IOCTL(11, AllocateTlbPayload, "out", _size=1 << 21)
FreeTlb = _TT_IOCTL(12, FreeTlbPayload)
ConfigureTlb = _TT_IOCTL(13, ConfigureTlbPayload)
SetPowerState = _TT_IOCTL(15, PowerState, _argsz=ctypes.sizeof(PowerState), _validity=4)

class Sysmem:
  PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

  def __init__(self, fd:FileIOInterface, size:int=1 << 30):
    self.fd, self.size = fd, (size + self.PAGE_SIZE - 1) & -self.PAGE_SIZE
    self.allocator = BumpAllocator(self.size, wrap=False)
    addr = FileIOInterface.anon_mmap(0, self.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, 0)
    self.view:MMIOInterface|None = MMIOInterface(addr, self.size)
    try: self.noc_addr:int|None = PinPages(fd, virtual_address=addr, size=self.size).noc_address
    except Exception:
      FileIOInterface.munmap(addr, self.size)
      self.view = None
      raise

  @property
  def addr(self) -> int:
    assert self.view is not None, "sysmem is closed"
    return self.view.addr

  def alloc(self, size:int, alignment:int|None=None) -> int:
    return self.allocator.alloc(size, self.PAGE_SIZE if alignment is None else alignment)

  def read(self, offset:int, size:int) -> bytes:
    assert self.view is not None, "sysmem is closed"
    return bytes(self.view[offset:offset+size])

  def write(self, offset:int, data:bytes|bytearray|memoryview):
    assert self.view is not None, "sysmem is closed"
    self.view[offset:offset+len(data)] = data

  def close(self):
    if self.view is None: return
    if self.noc_addr is not None:
      UnpinPages(self.fd, virtual_address=self.view.addr, size=self.size)
      self.noc_addr = None
    if FileIOInterface.munmap(self.view.addr, self.size) != 0: raise OSError(ctypes.get_errno(), "munmap sysmem failed")
    self.view = None

  @suppress_finalizing
  def __del__(self): self.close()

class TLBWindow:
  SIZE = 1 << 21
  USER_ID_LIMIT = 201
  WORKER_START = (1, 2)
  WORKER_END = (14, 11)

  def __init__(self, fd:FileIOInterface, core:tuple[int, int]):
    tlb = AllocateTlb(fd)
    self.fd, self.id, self.core = fd, tlb.id, core
    if self.id >= self.USER_ID_LIMIT:
      FreeTlb(fd, id=self.id)
      raise RuntimeError(f"driver returned reserved TLB id {self.id}")
    try:
      addr = fd.mmap(0, self.SIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, tlb.mmap_offset_uc)
      self.view:MMIOInterface|None = MMIOInterface(addr, self.SIZE)
    except Exception:
      FreeTlb(fd, id=self.id)
      self.id, self.view = None, None
      raise

  @property
  def addr(self) -> int:
    assert self.view is not None, "TLB window is closed"
    return self.view.addr

  def target(self, addr:int, start:tuple[int, int]|None=None, end:tuple[int, int]|None=None):
    ConfigureTlb(self.fd, id=self.id, addr=addr, start=self.core if start is None else start, end=end)

  def read(self, offset:int, bytes:int=4) -> builtins.bytes:  # noqa: A002
    assert self.view is not None, "TLB window is closed"
    return builtins.bytes(self.view[offset:offset+bytes])

  def write(self, offset:int, value:int|bytes|bytearray|memoryview, bytes:int=4):  # noqa: A002
    assert self.view is not None, "TLB window is closed"
    data = value.to_bytes(bytes, "little") if isinstance(value, int) else value
    self.view[offset:offset+len(data)] = data

  def mcast(self, addr:int, value:int|bytes|bytearray|memoryview, bytes:int=4):  # noqa: A002
    base = addr & -self.SIZE
    self.target(base, self.WORKER_START, self.WORKER_END)
    self.write(addr - base, value, bytes)

  def close(self):
    if self.view is not None:
      if FileIOInterface.munmap(self.view.addr, self.SIZE) != 0: raise OSError(ctypes.get_errno(), "munmap TLB failed")
      self.view = None
    if self.id is not None:
      FreeTlb(self.fd, id=self.id)
      self.id = None

  @suppress_finalizing
  def __del__(self): self.close()

  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): self.close()

class PCIDevice:
  P100A_X = (*range(1, 8), *range(10, 15))
  prefetch_core = (14, 2)
  dispatch_core = (14, 3)
  dram_core = (14, 4)

  def __init__(self, index:int=0, sysmem_size:int=1 << 30):
    card_type = FileIOInterface(f"/sys/class/tenstorrent/tenstorrent!{index}/tt_card_type").read().strip()
    if card_type != "p100a": raise RuntimeError(f"unsupported Blackhole card {card_type}; only p100a is supported")

    self.fd:FileIOInterface|None = FileIOInterface(f"/dev/tenstorrent/{index}", os.O_RDWR | os.O_CLOEXEC | os.O_APPEND)
    self._powered, self.sysmem = False, None
    try:
      SetPowerState(self.fd, power_flags=0b1111)
      self._powered = True
      self.dram_endpoints, self.cores = P100_DRAM_ENDPOINTS, list(P100_WORKER_CORES)
      self.sysmem = Sysmem(self.fd, sysmem_size)
    except Exception:
      self.close()
      raise

  def close(self):
    if self.fd is None: return
    if getattr(self, "sysmem", None) is not None:
      assert self.sysmem is not None
      self.sysmem.close()
      self.sysmem = None
    if self._powered:
      SetPowerState(self.fd, power_flags=0)
      self._powered = False
    os.close(self.fd.fd)
    del self.fd.fd
    self.fd = None

  @suppress_finalizing
  def __del__(self): self.close()
