import os, unittest

from tinygrad import Device
from tinygrad.device import BufferSpec, TinyELF
from tinygrad.helpers import Target
from tinygrad.renderer.tensix import TensixRenderer
from tinygrad.runtime.ops_tt import TTDevice
from tinygrad.runtime.support.tt.asm import Asm
from tinygrad.runtime.support.tt.firmware.consts import KERNEL_ROLES, TensixL1
from tinygrad.runtime.support.tt.pcie import P100_DRAM_ENDPOINTS
from tinygrad.runtime.support.tt.program import decode_tt_program, encode_tt_program


class TestTTProgramContainer(unittest.TestCase):
  def test_backend_registration(self): self.assertIs(Device.get_class("TT"), TTDevice)

  def test_launch_model_has_no_gpu_thread_groups(self):
    self.assertFalse(TensixRenderer.has_local)
    self.assertFalse(TensixRenderer.has_threads)

  def test_roundtrip(self):
    images = {"brisc": b"\x01\x02\x03\x04", "trisc1": b"math"}
    decoded = decode_tt_program(encode_tt_program(images))
    self.assertEqual(decoded, {role: images.get(role, b"") for role in KERNEL_ROLES})

  def test_rejects_trailing_bytes(self):
    with self.assertRaises(ValueError): decode_tt_program(encode_tt_program({}) + b"bad")


@unittest.skipUnless(os.getenv("TT_PROGRAM_TEST") == "1", "set TT_PROGRAM_TEST=1 and run through tt-device-queue")
class TestTTProgramHardware(unittest.TestCase):
  def test_hcq_launch_and_copy(self):
    value = 0xC0DEC0DE
    kernel = Asm("brisc")
    with kernel.scope():
      target = kernel.reg()
      kernel.read(target, TensixL1.PARAM_BASE)
      scratch = TensixL1.DATA_BUFFER_SPACE_BASE
      kernel.write(scratch, value)
      kernel.noc.write(scratch, target, kernel.noc.coordinate(*P100_DRAM_ENDPOINTS[0][0]), 4, posted=False)

    dev = TTDevice(f"TT:{os.getenv('TT_DEVICE_INDEX', '0')}")
    spec = BufferSpec(nolru=True)
    output = dev.allocator.alloc(4, spec)
    try:
      dev.allocator._copyin(output, memoryview(bytearray(4)))
      obj = TinyELF(encode_tt_program({"brisc": kernel.lower()}), "write_constant", Target("TT", arch="blackhole"), ())
      dev.runtime(obj)(output, wait=True)
      result = memoryview(bytearray(4))
      dev.allocator._copyout(result, output)
      self.assertEqual(int.from_bytes(result, "little"), value)
    finally:
      dev.allocator.free(output, 4, spec)
      dev.finalize()


if __name__ == "__main__": unittest.main()
