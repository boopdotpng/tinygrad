import contextlib, io, os, unittest
from dataclasses import replace

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec, TinyELF
from tinygrad.helpers import Target
from tinygrad.renderer.tensix import TensixRenderer
from tinygrad.runtime.ops_tt import TTDevice
from tinygrad.runtime.support.tt.asm import Asm
from tinygrad.runtime.support.tt.cq import DramCopy
from tinygrad.runtime.support.tt.firmware.consts import KERNEL_ROLES, TensixL1
from tinygrad.runtime.support.tt.program import decode_tt_program, encode_tt_program
from tinygrad.uop.ops import Ops, UOp


class TestTTProgramContainer(unittest.TestCase):
  def test_backend_registration(self): self.assertIs(Device.get_class("TT"), TTDevice)

  def test_launch_model_has_no_gpu_thread_groups(self):
    self.assertFalse(TensixRenderer.has_local)
    self.assertFalse(TensixRenderer.has_threads)

  def test_placeholder_renderer_dumps_uops_and_emits_valid_program(self):
    renderer = TensixRenderer(Target("TT", arch="blackhole"))
    with contextlib.redirect_stdout(io.StringIO()): source = renderer.render([UOp.const(7, dtypes.weakint)])
    self.assertIn("CONST", source)
    self.assertEqual(decode_tt_program(renderer.compiler.compile(source)), {role:b"" for role in KERNEL_ROLES})

  def test_placeholder_tensor_codegen(self):
    out = (Tensor.empty(4, device="TT", dtype=dtypes.float) + 1).cast(dtypes.float)
    linear = out.schedule_linear()
    ast = next(call.src[0] for call in linear.src if call.op is Ops.CALL and call.src[0].op is Ops.SINK)
    # TT must ignore both BEAM and the generic hand-coded GPU optimizations.
    ast = ast.replace(arg=replace(ast.arg, beam=4))
    with contextlib.redirect_stdout(io.StringIO()): prg = to_program(ast, TensixRenderer(Target("TT", arch="blackhole")))
    self.assertIn("ADD", prg.src[2].arg)
    self.assertEqual(prg.src[0].arg.applied_opts, ())
    self.assertEqual(sum(u.op is Ops.STORE for u in prg.src[1].src), 1)
    self.assertEqual(decode_tt_program(prg.src[3].arg), {role:b"" for role in KERNEL_ROLES})

  def test_roundtrip(self):
    images = {"brisc": b"\x01\x02\x03\x04", "trisc1": b"math"}
    decoded = decode_tt_program(encode_tt_program(images))
    self.assertEqual(decoded, {role: images.get(role, b"") for role in KERNEL_ROLES})

  def test_rejects_trailing_bytes(self):
    with self.assertRaises(ValueError): decode_tt_program(encode_tt_program({}) + b"bad")

  def test_dram_copy_accepts_p150_bank_seven(self):
    self.assertEqual(len(DramCopy(0x40, 0x1000, 16, 1, 1, bank_start=7).lower()), 64)


@unittest.skipUnless(os.getenv("TT_PROGRAM_TEST") == "1", "set TT_PROGRAM_TEST=1 and run through tt-device-queue")
class TestTTProgramHardware(unittest.TestCase):
  def test_hcq_launch_and_copy(self):
    value = 0xC0DEC0DE
    dev = TTDevice(f"TT:{os.getenv('TT_DEVICE_INDEX', '0')}")
    spec = BufferSpec(nolru=True)
    output = dev.allocator.alloc(4, spec)
    try:
      kernel = Asm("brisc")
      with kernel.scope():
        target = kernel.reg()
        kernel.read(target, TensixL1.PARAM_BASE)
        scratch = TensixL1.DATA_BUFFER_SPACE_BASE
        kernel.write(scratch, value)
        kernel.noc.write(scratch, target, kernel.noc.coordinate(*dev.iface.pcie.dram_endpoints[0][0]), 4, posted=False)

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
