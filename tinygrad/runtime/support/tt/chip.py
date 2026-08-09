from __future__ import annotations
import time
from typing import TYPE_CHECKING

from tinygrad.runtime.support.tt.cq import DRAM_BRISC_READY, DRAM_NCRISC_READY
from tinygrad.runtime.support.tt.firmware.consts import Firmware, FirmwareControl, RunState, TensixL1, TensixMMIO
from tinygrad.runtime.support.tt.firmware.core import build_brisc, build_ncrisc, build_trisc
from tinygrad.runtime.support.tt.firmware.cq import build_dispatch, build_prefetch
from tinygrad.runtime.support.tt.firmware.dram_cq import build_dram_brisc, build_dram_ncrisc
from tinygrad.runtime.support.tt.isa import R, RV32
from tinygrad.runtime.support.tt.pcie import PCIDevice, TLBWindow

if TYPE_CHECKING:
  from tinygrad.runtime.support.tt.cq import CommandQueue


class TTChip:
  """Blackhole boot and resident-firmware owner.

  PCIe mappings and ioctls stay in pcie.py. Host queue packets stay in cq.py.
  This class owns only chip reset, firmware construction/upload, and core start.
  """

  def __init__(self, pcie:PCIDevice):
    self.pcie, self.booted = pcie, False

  def _fd(self):
    if self.pcie.fd is None: raise RuntimeError("TT PCIe device is closed")
    return self.pcie.fd

  def reset_cores(self):
    with TLBWindow(self._fd(), self.pcie.cores[0], self.pcie.worker_end) as win:
      win.mcast(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TensixMMIO.SOFT_RESET_ALL)
    self.booted = False

  @staticmethod
  def _worker_firmware() -> bytes:
    images = (build_brisc(), build_ncrisc(), *(build_trisc(i) for i in range(3)))
    lowered = tuple(image.lower() for image in images)
    for (role, (_, capacity)), image in zip(Firmware.TEXT.items(), lowered):
      if len(image) > capacity: raise RuntimeError(f"{role} firmware exceeds its L1 text partition")
    return b"".join(image.ljust(size, b"\0") for (_, size), image in zip(Firmware.TEXT.values(), lowered))

  def boot(self, queue:CommandQueue, timeout:float=5.0):
    """Reset all Tensix cores and boot worker plus queue firmware.

    CommandQueue must be constructed first so its pinned sysmem regions and L1
    queue mailboxes exist before the resident queue cores receive GO.
    """
    if queue.pcie is not self.pcie: raise ValueError("command queue belongs to a different TT PCIe device")

    firmware = self._worker_firmware()
    firmware_base = Firmware.TEXT["brisc"][0]
    prefetch = build_prefetch(queue.pcie_mid).lower()
    dispatch = build_dispatch(queue.pcie_mid).lower()
    dram_brisc = build_dram_brisc(self.pcie.dram_endpoints).lower()
    dram_ncrisc = build_dram_ncrisc(self.pcie.dram_endpoints).lower()
    queue_images = (
      (self.pcie.prefetch_core, {"brisc": prefetch}),
      (self.pcie.dispatch_core, {"brisc": dispatch}),
      (self.pcie.dram_core, {"brisc": dram_brisc, "ncrisc": dram_ncrisc}),
    )
    for _, images in queue_images:
      for role, image in images.items():
        if len(image) > TensixL1.WORKER_TEXT_SIZE[role]:
          raise RuntimeError(f"queue {role} firmware exceeds its L1 text partition")

    try:
      with TLBWindow(self._fd(), self.pcie.cores[0], self.pcie.worker_end) as win:
        win.mcast(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TensixMMIO.SOFT_RESET_ALL)
        win.mcast(firmware_base, firmware)
        win.mcast(TensixL1.BOOT, RV32().jal(R.ZERO, firmware_base + 4).to_bytes(4, "little"))
        win.mcast(FirmwareControl.GO_SIGNAL & -4, 0)
        win.mcast(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)

        for core, images in queue_images:
          win.target(0, core)
          for role, image in images.items(): win.write(TensixL1.WORKER_TEXT_BASE[role], image)

        win.target(0, self.pcie.dram_core)
        win.write(DRAM_BRISC_READY, bytes(8))
        for core, _ in queue_images:
          win.target(0, core)
          win.write(FirmwareControl.GO_SIGNAL, int(RunState.GO), bytes=1)

        win.target(0, self.pcie.dram_core)
        deadline = time.monotonic() + timeout
        while (int.from_bytes(win.read(DRAM_BRISC_READY), "little") != 1 or
               int.from_bytes(win.read(DRAM_NCRISC_READY), "little") != 1):
          if time.monotonic() >= deadline: raise TimeoutError("TT command-queue DRAM engines did not start")
          time.sleep(0)
      self.booted = True
    except Exception:
      self.reset_cores()
      raise

  def fini(self):
    if self.pcie.fd is not None: self.reset_cores()
