import ctypes, os, unittest

from tinygrad.runtime.support.tt.pcie import (
  AllocateTlbIn, AllocateTlbOut, AllocateTlbPayload, ConfigureTlbIn, ConfigureTlbPayload, FreeTlbPayload, NocTlbConfig,
  P100_DRAM_ENDPOINTS, P100_WORKER_CORES, P150_DRAM_ENDPOINTS, P150_WORKER_CORES, PCIDevice, PinPagesIn, PinPagesOut,
  PinPagesPayload, PowerState, TLBWindow, UnpinPagesPayload, board_config,
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

class TestTTBoardConfig(unittest.TestCase):
  def test_p100a(self):
    config = board_config("p100a", 0xFFF, 0x7F)
    self.assertEqual(config.cores, P100_WORKER_CORES)
    self.assertEqual(config.dram_endpoints, P100_DRAM_ENDPOINTS)
    self.assertEqual((config.prefetch_core, config.dispatch_core, config.dram_core), ((14, 2), (14, 3), (14, 4)))
    self.assertEqual(config.worker_end, (14, 11))

  def test_p150a_stock_topology(self):
    config = board_config("p150a", 0xFFF, 0xFF)
    self.assertEqual(config.cores, P100_WORKER_CORES)
    self.assertEqual(config.dram_endpoints, P150_DRAM_ENDPOINTS)
    self.assertEqual((config.prefetch_core, config.dispatch_core, config.dram_core), ((14, 2), (14, 3), (14, 4)))
    self.assertEqual(config.worker_end, (14, 11))

  def test_p150a_restored_topology(self):
    config = board_config("p150a", 0x3FFF, 0xFF)
    self.assertEqual(config.cores, P150_WORKER_CORES)
    self.assertEqual(config.dram_endpoints, P150_DRAM_ENDPOINTS)
    self.assertEqual((config.prefetch_core, config.dispatch_core, config.dram_core), ((16, 2), (16, 3), (16, 4)))
    self.assertEqual(config.worker_end, (16, 11))

  def test_rejects_bad_topologies(self):
    for args in (("p100a", 0xFFF, 0xFF), ("p150a", 0xFFF, 0x7F), ("unknown", 0xFFF, 0x7F)):
      with self.subTest(args=args), self.assertRaises(RuntimeError): board_config(*args)

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
