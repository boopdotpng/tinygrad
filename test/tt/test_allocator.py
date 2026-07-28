import os, unittest

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_tt import DRAM_BANKS, DRAM_PAGE_SIZE, DRAM_START, TTAllocation, TTAllocator, TTDevice, TT_VA_SIZE, TT_VA_START
from tinygrad.runtime.support.hcq import HCQBuffer
from tinygrad.runtime.support.memory import TLSFAllocator


class TestTTAllocator(unittest.TestCase):
  @staticmethod
  def _dram_allocator():
    alloc = object.__new__(TTAllocator)
    alloc.dev = object()
    alloc.dram_mm = TLSFAllocator((1 << 32) - DRAM_START, base=DRAM_START, block_size=64)
    alloc.va_mm = TLSFAllocator(TT_VA_SIZE, base=TT_VA_START, block_size=64)
    return alloc

  def test_exact_visible_size(self):
    alloc = self._dram_allocator()
    one = alloc._alloc(1, BufferSpec())
    large = alloc._alloc(DRAM_PAGE_SIZE * DRAM_BANKS + 3, BufferSpec())
    self.assertEqual((one.size, one.meta.size), (1, 1))
    self.assertEqual((large.size, large.meta.size), (DRAM_PAGE_SIZE * DRAM_BANKS + 3,) * 2)
    self.assertEqual(one.meta.capacity, DRAM_PAGE_SIZE)
    self.assertEqual(large.meta.capacity, 2 * DRAM_PAGE_SIZE)
    self.assertNotEqual(one.va_addr, large.va_addr)
    self.assertGreaterEqual(large.meta.dram_addr, one.meta.dram_addr + one.meta.capacity)

  def test_dense_pages_rotate_over_banks(self):
    size = DRAM_PAGE_SIZE * (DRAM_BANKS + 1) + 3
    dram_meta = TTAllocation("dram", TT_VA_START, size, 2 * DRAM_PAGE_SIZE, dram_addr=0x1000)
    host_meta = TTAllocation("host", 0x8000_0000, size, size + 13, sysmem_offset=0)
    dram = HCQBuffer(dram_meta.va_addr, size, meta=dram_meta)
    host = HCQBuffer(host_meta.va_addr, size, meta=host_meta)

    commands = TTAllocator.copy_commands(dram, host, size)
    self.assertEqual(len(commands), DRAM_BANKS + 2)
    self.assertEqual([command.bank_start for command in commands], [*range(DRAM_BANKS), 0, 1])
    self.assertEqual([command.addr for command in commands[:DRAM_BANKS]], [0x1000] * DRAM_BANKS)
    self.assertEqual(commands[DRAM_BANKS].addr, 0x1000 + DRAM_PAGE_SIZE)
    self.assertEqual(commands[-1].page_size, 16)  # transport bounce only; visible size remains exact

  def test_short_interior_copy_is_rejected(self):
    dram_meta = TTAllocation("dram", TT_VA_START, 128, DRAM_PAGE_SIZE, dram_addr=0x1000)
    host_meta = TTAllocation("host", 0x8000_0000, 128, 128, sysmem_offset=0)
    dram = HCQBuffer(dram_meta.va_addr, 128, meta=dram_meta)
    host = HCQBuffer(host_meta.va_addr, 128, meta=host_meta)
    with self.assertRaises(ValueError): TTAllocator.copy_commands(dram, host, 3)


@unittest.skipUnless(os.getenv("TT_ALLOCATOR_TEST") == "1", "set TT_ALLOCATOR_TEST=1 and run through tt-device-queue")
class TestTTAllocatorHardware(unittest.TestCase):
  def test_exact_size_roundtrip(self):
    dev = TTDevice(f"TT:{os.getenv('TT_DEVICE_INDEX', '0')}")
    try:
      for size in (1, 15, 16, 17, 4095, 4096, 4097, 7 * 4096 + 3):
        with self.subTest(size=size):
          spec = BufferSpec(nolru=True)
          buf = dev.allocator.alloc(size, spec)
          try:
            source = memoryview(bytearray((i * 29 + 7) & 0xFF for i in range(size)))
            dev.allocator._copyin(buf, source)
            result = memoryview(bytearray(size))
            dev.allocator._copyout(result, buf)
            self.assertEqual(result, source)
          finally: dev.allocator.free(buf, size, spec)
    finally: dev.finalize()


if __name__ == "__main__": unittest.main()
