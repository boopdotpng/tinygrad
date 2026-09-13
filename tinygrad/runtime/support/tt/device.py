from __future__ import annotations
import fcntl, hashlib, struct, time
from pathlib import Path
from tinygrad.helpers import DEBUG, fetch, getenv, unwrap
from tinygrad.runtime.support.tt.pcie import PCIDevice, TLBWindow
from tinygrad.runtime.support.tt.consts import Firmware, FirmwareControl, RunState, TensixL1, TensixMMIO

FW_URL = 'https://github.com/boopdotpng/blackhole-py/releases/download/hcq-v1/bh_hcq_v1.bin'
FW_SHA256 = '9b29e945fdd804a0fc405375f9913d12c67547457c78b4f15edc0a9596da69e4'

def firmware() -> tuple[bytes, ...]:
  blob = Path(p).read_bytes() if (p:=getenv('TT_FIRMWARE', '')) else fetch(FW_URL, subdir='fw', sha256=FW_SHA256).read_bytes()
  if not p and hashlib.sha256(blob).hexdigest() != FW_SHA256: raise ValueError('firmware checksum mismatch')
  if len(blob) < 44 or blob[:8] != b'BHCQ0001': raise ValueError('unsupported Blackhole firmware ABI')
  sizes, offset, images = struct.unpack_from('<9I', blob, 8), 44, []
  if not all(sizes) or 44 + sum(sizes) != len(blob): raise ValueError('invalid firmware image sizes')
  limits = [s for _, s in Firmware.TEXT.values()] + [TensixL1.WORKER_TEXT_SIZE[r] for r in ('brisc','brisc','brisc','ncrisc')]
  for size, limit in zip(sizes, limits):
    if size > limit: raise ValueError('firmware exceeds L1 image slot')
    images.append(blob[offset:offset+size])
    offset += size
  return tuple(images)

def jal(offset:int) -> bytes:
  if offset % 2 or not -(1<<20) <= offset < (1<<20): raise ValueError('invalid JAL offset')
  x = offset & 0x1fffff
  return struct.pack('<I', (x>>20)<<31 | ((x>>1)&1023)<<21 | ((x>>11)&1)<<20 | ((x>>12)&255)<<12 | 0x6f)

class TTInterface:
  def __init__(self, index:int):
    self.images, self.peer_group = firmware(), f'TT:{index}'
    self.lock = open(f'/tmp/blackhole-py-raw-device-{index}.lock', 'a')
    try:
      fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      self.pcie = PCIDevice(index)
      self.sysmem = unwrap(self.pcie.sysmem)
      if self.sysmem.noc_addr >> 32 != (self.sysmem.noc_addr+self.sysmem.size-1) >> 32: raise RuntimeError('sysmem crosses a 4 GiB NoC aperture')
      self.prefetch = TLBWindow(self.pcie.fd, self.pcie.prefetch_core)
      self.prefetch.target(0)
    except Exception:
      if hasattr(self,'prefetch'): self.prefetch.close()
      if hasattr(self,'pcie'): self.pcie.close()
      self.lock.close()
      raise

  def boot(self, issue:int, read_ptr:int):
    boot_start = time.perf_counter()
    images = self.images
    if DEBUG >= 2: print(f"{self.peer_group}: loading firmware ({sum(map(len, images))} bytes)")
    resident = b''.join(img.ljust(size, b'\0') for (_, size), img in zip(Firmware.TEXT.values(), images))
    with TLBWindow(self.pcie.fd, self.pcie.cores[0]) as w:
      def broadcast(addr, value):
        w.target(addr & -w.SIZE, (1,2), (14,11))
        w.write(addr % w.SIZE, value)
      broadcast(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TensixMMIO.SOFT_RESET_ALL)
      broadcast(Firmware.TEXT['brisc'][0], resident)
      broadcast(0, jal(Firmware.TEXT['brisc'][0] + 4))
      broadcast(FirmwareControl.GO_SIGNAL & -4, 0)
      broadcast(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)
      for core, roles in ((self.pcie.prefetch_core, { 'brisc':images[5]}), (self.pcie.dispatch_core, {'brisc':images[6]}),
                          (self.pcie.dram_core, {'brisc':images[7], 'ncrisc':images[8]})):
        w.target(0, core)
        for role, image in roles.items(): w.write(TensixL1.WORKER_TEXT_BASE[role], image)
        w.write(0x1080, self.sysmem.noc_addr >> 32)
        w.write(0x1084, len(self.pcie.dram_endpoints))
        for niu in range(2):
          for bank, endpoints in enumerate(self.pcie.dram_endpoints):
            x, y = endpoints[niu]
            w.write(0x10a0 + niu*32 + bank*4, x | y<<6)
      w.target(0, self.pcie.dram_core)
      w.write(0x1008, bytes(8))
      w.target(0, self.pcie.dispatch_core)
      w.write(0x1000, 0)
      self.prefetch.write(0x1000, bytes(8))
      self.prefetch.write(0x1008, issue & 0xffffffff)
      self.prefetch.write(0x100c, read_ptr & 0xffffffff)
      self.prefetch.write(0x1010, 0)
      for core in (self.pcie.prefetch_core, self.pcie.dispatch_core, self.pcie.dram_core):
        w.target(0, core)
        w.write(FirmwareControl.GO_SIGNAL, int(RunState.GO), bytes=1)
      w.target(0, self.pcie.dram_core)
      start = time.monotonic()
      while w.read(0x1008,8) != struct.pack('<II',1,1):
        if time.monotonic()-start > 5: raise RuntimeError('DMA firmware boot timed out')
        time.sleep(0.001)
    if DEBUG >= 2: print(f"{self.peer_group}: firmware uploaded, CQ/DMA ready ({(time.perf_counter()-boot_start)*1000:.2f} ms)")

  def device_fini(self):
    if self.pcie.fd < 0: return
    with TLBWindow(self.pcie.fd, self.pcie.cores[0]) as w:
      addr = TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0
      w.target(addr & -w.SIZE, (1,2), (14,11))
      w.write(addr % w.SIZE, TensixMMIO.SOFT_RESET_ALL)
    self.prefetch.close()
    self.pcie.close()
    self.lock.close()
