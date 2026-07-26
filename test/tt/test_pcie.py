import ctypes, os, unittest

from tinygrad.runtime.support.tt.pcie import (
  AllocateTlbIn, AllocateTlbOut, AllocateTlbPayload, ConfigureTlbIn, ConfigureTlbPayload, FreeTlbPayload,
  NocTlbConfig, PCIDevice, PinPagesIn, PinPagesOut, PinPagesPayload, PowerState, TLBWindow, UnpinPagesPayload,
)

class TestTTPCIeABI(unittest.TestCase):
  def test_ioctl_struct_sizes(self):
    expected = {
      PinPagesIn: 24, PinPagesOut: 16, PinPagesPayload: 40, UnpinPagesPayload: 24,
      AllocateTlbIn: 16, AllocateTlbOut: 32, AllocateTlbPayload: 48, FreeTlbPayload: 4,
      NocTlbConfig: 32, ConfigureTlbIn: 40, ConfigureTlbPayload: 48, PowerState: 40,
    }
    for struct_t, size in expected.items():
      with self.subTest(struct_t=struct_t.__name__): self.assertEqual(ctypes.sizeof(struct_t), size)

  def test_tlb_config_layout(self):
    cfg = ConfigureTlbPayload(id=7, addr=0x12340000, start=(1, 2), end=(14, 11))
    self.assertEqual((cfg.id, cfg.addr), (7, 0x12340000))
    self.assertEqual((cfg.x_start, cfg.y_start, cfg.x_end, cfg.y_end), (1, 2, 14, 11))
    self.assertEqual((cfg._noc_mcast[0], cfg._noc_mcast[1], cfg._ordering), (0, 1, 1))

@unittest.skipUnless(os.getenv("TT_PCI_LOOPBACK") == "1", "set TT_PCI_LOOPBACK=1 and run through tt-device-queue")
class TestTTPCIeLoopback(unittest.TestCase):
  def test_l1_word(self):
    dev = PCIDevice(int(os.getenv("TT_DEVICE_INDEX", "0")), sysmem_size=2 << 20)
    address, value = 0x17FFFC, 0xC0DEC0DE
    try:
      assert dev.fd is not None
      with TLBWindow(dev.fd, dev.cores[0]) as win:
        base, offset = address & -win.SIZE, address & (win.SIZE - 1)
        win.target(base)
        previous = win.read(offset)
        try:
          win.write(offset, value)
          self.assertEqual(int.from_bytes(win.read(offset), "little"), value)
        finally:
          win.write(offset, previous)
          self.assertEqual(win.read(offset), previous)
    finally:
      dev.close()

if __name__ == "__main__": unittest.main()
