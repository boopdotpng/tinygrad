import unittest, struct
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.renderer.tt import TTRenderer, isa, pack_program, RET, PROG_HDR

class TestTTIsa(unittest.TestCase):
  def test_rv32(self):
    for ins, word in ((isa.addi(rd=5, rs1=0, imm=1), 0x00100293), (isa.lw(rd=5, rs1=3, imm=-16), 0xff01a283), (isa.jalr(rd=0, rs1=5), 0x00028067),
                      (isa.jal(imm=0x200), 0x2000006f), (isa.sw(rs1=6, rs2=28, imm=0), 0x01c32023), (isa.lui(rd=15, imm=0x100), 0x001007b7),
                      (isa.add(rd=28, rs1=28, rs2=10), 0x00ae0e33), (isa.bne(rs1=6, rs2=7, imm=-16), 0xfe7318e3),
                      (isa.mul(rd=1, rs1=2, rs2=3), 0x023100b3)):
      self.assertEqual(ins, word, f"{ins!r} is {ins:#010x}, expected {word:#010x}")

  def test_tensix(self): # the words the worker firmware resets a core with
    for ins, word in ((isa.ZEROACC(mode=3), 0x10180000), (isa.SFPENCC(imm=3, mode=10), 0x8a00300a), (isa.NOP(), 0x02000000),
                      (isa.SFPLOADI(imm=0xbf80), 0x7100bf80), (isa.SFPCONFIG(target=isa.LReg.NEG_ONE), 0x910000b0),
                      (isa.SEMINIT(max=1, sems=0xff), 0xa31003fc), (isa.SETRWC(mask=15), 0x3700000f), (isa.SETADCXY(counters=7, mask=15), 0x51e0000f),
                      (isa.SETC16(word=3, value=0x1234), 0xb2031234)):
      self.assertEqual(ins, word, f"{ins!r} is {ins:#010x}, expected {word:#010x}")
      self.assertEqual(isa.decode(word), ins)
      self.assertEqual(isa.outline(isa.inline(word)), word)
    self.assertEqual(isa.inline(0x8a00300a), 0x2800c02a) # a TRISC runs a Tensix instruction from its own stream rotated left by two

  def test_fields(self):
    self.assertEqual((ins:=isa.bne(rs1=6, rs2=7, imm=-16)).fields(), {"rs1": 6, "rs2": 7, "imm": -16, "opcode": 0x63, "funct3": 1})
    self.assertEqual(repr(ins), "bne(rs1=6, rs2=7, imm=-16)")
    self.assertEqual(repr(isa.SFPLOADI(lreg=1, imm=3)), "SFPLOADI(lreg=L1, imm=3)")
    for bad in (lambda: isa.addi(rd=1, rs1=0, imm=2048), lambda: isa.bne(rs1=0, rs2=0, imm=3), lambda: isa.add(rd=32), lambda: isa.NOP(1)):
      with self.assertRaises((ValueError, TypeError)): bad()

class TestTTProgram(unittest.TestCase):
  def test_pack_program(self):
    blob = pack_program({"brisc": b"\x13\0\0\0", "trisc1": b"\x13\0\0\0" * 3}, cb_mask=3)
    self.assertEqual(struct.unpack_from("<6I", blob), (PROG_HDR, 0, 0, PROG_HDR + 16, 0, 3))
    self.assertEqual(blob[PROG_HDR:PROG_HDR + 12], b"\x13\0\0\0" + RET) # the code returns to the firmware
    self.assertEqual(blob[PROG_HDR + 28:PROG_HDR + 36], RET)
    self.assertEqual(len(blob) % 64, 0)
    with self.assertRaises(AssertionError): pack_program({"gpu": b""})
    with self.assertRaises(RuntimeError): pack_program({"brisc": bytes(1 << 20)})

  def test_asm(self):
    insts = [isa.addi(rd=5, rs1=0, imm=1), "trisc1", isa.SFPLOADI(imm=0xbf80), isa.addi(rd=6, rs1=0, imm=2), "brisc", isa.jalr(rd=0, rs1=1)]
    blob = TTRenderer(Target("TT", arch="p150a")).asm(UOp(Ops.PROGRAM), UOp(Ops.LINEAR, src=tuple(UOp.native(x).call() for x in insts)))
    self.assertEqual(struct.unpack_from("<5I", blob), (PROG_HDR, 0, 0, PROG_HDR + 16, 0))
    self.assertEqual(struct.unpack_from("<2I", blob, PROG_HDR), (0x00100293, 0x00008067))
    self.assertEqual(struct.unpack_from("<2I", blob, PROG_HDR + 16), (isa.inline(0x7100bf80), 0x00200313))

  def test_no_lowering(self):
    out, inp, i = UOp.param(0, dtypes.int, 4), UOp.param(1, dtypes.int, 4), UOp.range(4, 0)
    ast = out[i].store(inp[i] + 1).end(i).sink(arg=KernelInfo(name="add_one"))
    with self.assertRaises(NotImplementedError): to_program(ast, TTRenderer(Target("TT", arch="p150a")))

if __name__ == "__main__":
  unittest.main()
