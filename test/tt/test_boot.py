import os, unittest

from tinygrad.runtime.support.tt.chip import TTChip
from tinygrad.runtime.support.tt.cq import CommandQueue
from tinygrad.runtime.support.tt.pcie import PCIDevice


@unittest.skipUnless(os.getenv("TT_BOOT_TEST") == "1", "set TT_BOOT_TEST=1 and run through tt-device-queue")
class TestTTBoot(unittest.TestCase):
  def test_boot_resident_firmware(self):
    pcie = PCIDevice(int(os.getenv("TT_DEVICE_INDEX", "0")))
    queue = chip = None
    try:
      queue = CommandQueue(pcie)
      chip = TTChip(pcie)
      chip.boot(queue)
      self.assertTrue(chip.booted)
    finally:
      if chip is not None: chip.fini()
      if queue is not None: queue.close()
      pcie.close()


if __name__ == "__main__": unittest.main()
