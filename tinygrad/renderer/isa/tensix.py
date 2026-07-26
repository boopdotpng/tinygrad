# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
import struct
from tinygrad.uop import FastEnum, auto, Ops
from tinygrad.uop.ops import UOp
from tinygrad.renderer.isa import Register, greg

# ***** RV32 Ops *****

class RV32Ops(FastEnum):
  # not real instructions: LABEL marks a code location, DEFINE reads an existing register without emitting an instruction
  LABEL = auto(); DEFINE = auto()
  # R-type int
  ADD = auto(); SUB = auto(); SLTU = auto(); XOR = auto(); OR = auto(); AND = auto()
  # R-type M extension
  MUL = auto(); DIVU = auto(); REMU = auto()
  # I-type int
  ADDI = auto(); SLTIU = auto(); XORI = auto(); ORI = auto(); ANDI = auto(); SLLI = auto(); SRLI = auto(); SRAI = auto()
  # loads / stores
  LW = auto(); LBU = auto(); LHU = auto()
  SB = auto(); SH = auto(); SW = auto()
  # branches / jumps
  BEQ = auto(); BNE = auto(); BLT = auto(); BGE = auto(); BLTU = auto(); BGEU = auto()
  JAL = auto(); JALR = auto()
  # upper immediate
  LUI = auto(); AUIPC = auto()
  # system subset currently used by blackhole-py
  CSRRS = auto(); CSRRC = auto(); FENCE = auto()

# ***** Tensix Ops *****

class TensixOps(FastEnum):
  ADDDMAREG = auto(); ATGETM = auto(); ATRELM = auto(); DMANOP = auto()
  ELWADD = auto(); ELWMUL = auto(); INCRWC = auto(); MOP = auto()
  MOVA2D = auto(); MULDMAREG = auto(); MVMUL = auto(); NOP = auto()
  PACR = auto(); RDCFG = auto(); REPLAY = auto(); RMWCIB0 = auto()
  RMWCIB1 = auto(); RMWCIB2 = auto(); RMWCIB3 = auto(); SEMGET = auto()
  SEMINIT = auto(); SEMPOST = auto(); SEMWAIT = auto(); SETADC = auto()
  SETADCXX = auto(); SETADCXY = auto(); SETADCZW = auto(); SETC16 = auto()
  SETDMAREG = auto(); SETRWC = auto(); SFPADD = auto(); SFPADDI = auto()
  SFPARECIP = auto(); SFPCAST = auto(); SFPCONFIG = auto(); SFPENCC = auto()
  SFPEXEXP = auto(); SFPEXMAN = auto(); SFPIADD = auto(); SFPLOAD = auto()
  SFPLOADI = auto(); SFPLUT = auto(); SFPMAD = auto(); SFPMOV = auto()
  SFPMUL = auto(); SFPNOP = auto(); SFPSETEXP = auto(); SFPSHFT = auto()
  SFPSHFT2 = auto(); SFPSTORE = auto(); SFPSWAP = auto(); STALLWAIT = auto()
  STOREREG = auto(); UNPACR = auto(); UNPACR_NOP = auto(); WRCFG = auto()
  ZEROACC = auto(); ZEROSRC = auto(); ADDRCRXY = auto(); ADDRCRZW = auto()
  APOOL3S1 = auto(); APOOL3S2 = auto(); ATCAS = auto(); ATINCGET = auto()
  ATINCGETPTR = auto(); ATSWAP = auto(); BITWOPDMAREG = auto(); CFGSHIFTMASK = auto()
  CLEARDVALID = auto(); CLREXPHIST = auto(); CMPDMAREG = auto(); CONV3S1 = auto()
  CONV3S2 = auto(); DOTPV = auto(); ELWSUB = auto(); FLUSHDMA = auto()
  GAPOOL = auto(); GATESRCRST = auto(); GMPOOL = auto(); INCADCXY = auto()
  INCADCZW = auto(); LOADIND = auto(); LOADREG = auto(); MFCONV3S1 = auto()
  MOP_CFG = auto(); MOVB2A = auto(); MOVB2D = auto(); MOVD2A = auto()
  MOVD2B = auto(); MOVDBGA2D = auto(); MOVDBGB2D = auto(); MPOOL3S1 = auto()
  MPOOL3S2 = auto(); PACR_SETREG = auto(); RAREB = auto(); REG2FLOP = auto()
  RESOURCEDECL = auto(); RSTDMA = auto(); SETASHRMH = auto(); SETASHRMH0 = auto()
  SETASHRMH1 = auto(); SETASHRMV = auto(); SETDVALID = auto(); SETIBRWC = auto()
  SETPKEDGOF = auto(); SFPABS = auto(); SFPAND = auto(); SFPCOMPC = auto()
  SFPDIVP2 = auto(); SFPGT = auto(); SFPLE = auto(); SFPLOADMACRO = auto()
  SFPLUTFP32 = auto(); SFPLZ = auto(); SFPMUL24 = auto(); SFPMULI = auto()
  SFPNOT = auto(); SFPOR = auto(); SFPPOPC = auto(); SFPPUSHC = auto()
  SFPSETCC = auto(); SFPSETMAN = auto(); SFPSETSGN = auto(); SFPSTOCHRND = auto()
  SFPTRANSP = auto(); SFPXOR = auto(); SHIFTDMAREG = auto(); SHIFTXA = auto()
  SHIFTXB = auto(); STOREIND = auto(); STREAMWAIT = auto(); STREAMWRCFG = auto()
  SUBDMAREG = auto(); TBUFCMD = auto(); TRNSPSRCA = auto(); TRNSPSRCB = auto()
  XMOV = auto()

# ***** RV32 registers *****
X = tuple(Register(f"x{i}", i, size=4) for i in range(32))
ZERO, RA, SP, GP = X[0], X[1], X[2], X[3]

WGPR = tuple(r for r in X if r not in (ZERO, RA, SP, GP))

reg_strs = {"x0": "zero", "x1": "ra", "x2": "sp", "x3": "gp", "x4": "tp", "x5": "t0", "x6": "t1", "x7": "t2",
            "x8": "s0", "x9": "s1", **{f"x{i}": f"a{i-10}" for i in range(10, 18)},
            **{f"x{i}": f"s{i-16}" for i in range(18, 28)}, **{f"x{i}": f"t{i-25}" for i in range(28, 32)}}

# ***** SFPU registers *****

# An LReg holds 32 lanes of 32 bits. Only L0-L7 are writable and therefore allocatable.
LREG = tuple(Register(f"l{i}", i, size=32*4) for i in range(8))
LREG_CONST_0_8373, LREG_ZERO, LREG_ONE, LREG_NEG_ONE, LREG_CONFIG0, LREG_CONFIG1, LREG_CONFIG2, LREG_LANE_X2 = \
  tuple(Register(name, i, size=32*4) for i,name in enumerate(
    ("const_0_8373", "zero", "one", "neg_one", "config0", "config1", "config2", "lane_x2"), start=8))
SFPU_FIXED_REGS = (LREG_CONST_0_8373, LREG_ZERO, LREG_ONE, LREG_NEG_ONE,
                   LREG_CONFIG0, LREG_CONFIG1, LREG_CONFIG2, LREG_LANE_X2)

# ***** instruction encoding *****

# https://github.com/riscv/riscv-isa-manual, RV32I base + M extension
def _signed(value:int, bits:int, alignment:int=1) -> int:
  if value % alignment or not -(1 << (bits-1)) <= value < (1 << (bits-1)): raise ValueError(f"{value} does not fit in a signed {bits}-bit immediate")
  return value & ((1 << bits) - 1)

def _unsigned(value:int, bits:int) -> int:
  if not 0 <= value < (1 << bits): raise ValueError(f"{value} does not fit in an unsigned {bits}-bit immediate")
  return value

def rv_r(op:int, f3:int, f7:int, rd:int, rs1:int, rs2:int) -> int: return f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
def rv_i(op:int, f3:int, rd:int, rs1:int, imm:int) -> int: return _signed(imm, 12) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
def rv_iu(op:int, f3:int, rd:int, rs1:int, imm:int) -> int: return _unsigned(imm, 12) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
def rv_s(op:int, f3:int, rs1:int, rs2:int, imm:int) -> int:
  imm = _signed(imm, 12)
  return (imm >> 5) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 0x1F) << 7 | op
def rv_b(op:int, f3:int, rs1:int, rs2:int, imm:int) -> int:
  imm = _signed(imm, 13, 2)
  return (imm >> 12) << 31 | ((imm >> 5) & 0x3F) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | ((imm >> 1) & 0xF) << 8 | ((imm >> 11) & 1) << 7 | op
def rv_u(op:int, rd:int, imm:int) -> int: return imm & 0xFFFFF000 | rd << 7 | op
def rv_j(op:int, rd:int, imm:int) -> int:
  imm = _signed(imm, 21, 2)
  return (imm >> 20) << 31 | ((imm >> 1) & 0x3FF) << 21 | ((imm >> 11) & 1) << 20 | ((imm >> 12) & 0xFF) << 12 | rd << 7 | op

OPC_OP, OPC_IMM, OPC_LOAD, OPC_STORE, OPC_BRANCH = 0x33, 0x13, 0x03, 0x23, 0x63
OPC_LUI, OPC_AUIPC, OPC_JAL, OPC_JALR, OPC_SYSTEM, OPC_FENCE = 0x37, 0x17, 0x6F, 0x67, 0x73, 0x0F

def _rd(x:UOp) -> int: return r.index if isinstance(r:=greg(x), Register) else 0
def _rs(x:UOp, i:int) -> int: return r.index if isinstance(r:=greg(x.src[i]), Register) else 0
def _imm(x:UOp, i:int) -> int:
  s = x.src[i]
  assert s.op is Ops.CONST, f"operand {i} of {x.arg} must be a constant, got {s.op}"
  return s.arg

# the uop shapes the isel matcher must produce. every RV32 operand is explicit in the encoding.
#   RType        dst=x                  src=(rs1, rs2)
#   IType/ShiftI dst=x                  src=(rs1, CONST imm)
#   Load         dst=x                  src=(rs1 base, CONST offset)
#   Store        dtype=void             src=(rs1 base, CONST offset, rs2 value)
#   Branch       dtype=void, tag=label  src=(rs1, rs2)
#   JAL          dtype=void, tag=label  (rd=x0; add a separate call form if linking is needed)
#   UType        dst=x                  src=(CONST imm)     NOTE: already shifted left 12, encoder masks the low bits
#   Csr          dst=x                  src=(rs1, CONST csr)

def _pack(word:int) -> bytes: return struct.pack("<I", word & 0xFFFFFFFF)
def enc_r(x:UOp, opc:int, f3:int, f7:int=0, disp:int=0) -> bytes: return _pack(rv_r(opc, f3, f7, _rd(x), _rs(x, 0), _rs(x, 1)))
def enc_i(x:UOp, opc:int, f3:int, disp:int=0) -> bytes: return _pack(rv_i(opc, f3, _rd(x), _rs(x, 0), _imm(x, 1)))
def enc_shift_i(x:UOp, opc:int, f3:int, f7:int=0, disp:int=0) -> bytes:
  return _pack(rv_i(opc, f3, _rd(x), _rs(x, 0), f7 << 5 | _unsigned(_imm(x, 1), 5)))
def enc_s(x:UOp, opc:int, f3:int, disp:int=0) -> bytes:
  # asm writes `sw rs2, off(rs1)` but the selected UOp keeps the base first.
  return _pack(rv_s(opc, f3, _rs(x, 0), _rs(x, 2), _imm(x, 1)))
def enc_b(x:UOp, opc:int, f3:int, disp:int=0) -> bytes: return _pack(rv_b(opc, f3, _rs(x, 0), _rs(x, 1), disp))
def enc_u(x:UOp, opc:int, disp:int=0) -> bytes: return _pack(rv_u(opc, _rd(x), _imm(x, 0)))
def enc_j(x:UOp, opc:int, disp:int=0) -> bytes: return _pack(rv_j(opc, _rd(x), disp))
def enc_csr(x:UOp, opc:int, f3:int, disp:int=0) -> bytes: return _pack(rv_iu(opc, f3, _rd(x), _rs(x, 0), _imm(x, 1)))

def _tt_field(x:UOp, i:int) -> int:
  s = x.src[i]
  if isinstance(r:=greg(s), Register): return r.index
  assert s.op is Ops.CONST, f"field {i} of {x.arg} must be a constant or register, got {s.op}"
  return s.arg

def tt_word(opcode:int, fields:tuple[tuple[int, int], ...], values:tuple[int, ...]) -> int:
  if len(fields) != len(values): raise ValueError(f"expected {len(fields)} Tensix fields, got {len(values)}")
  word = _unsigned(opcode, 8) << 24
  for value,(hi,lo) in zip(values, fields):
    word |= _unsigned(value, hi-lo+1) << lo
  if word >= 0xC0000000: raise ValueError(f"Tensix instruction looks like RISC-V: 0x{word:08x}")
  return word

def enc_tt(x:UOp, opcode:int, fields:tuple[tuple[int, int], ...], disp:int=0) -> bytes:
  return _pack(tt_word(opcode, fields, tuple(_tt_field(x, i) for i in range(len(x.src)))))

encodings = {
  # R-type int
  RV32Ops.ADD: lambda x, disp=0: enc_r(x, OPC_OP, 0, 0x00), RV32Ops.SUB: lambda x, disp=0: enc_r(x, OPC_OP, 0, 0x20),
  RV32Ops.SLTU: lambda x, disp=0: enc_r(x, OPC_OP, 3, 0x00), RV32Ops.XOR: lambda x, disp=0: enc_r(x, OPC_OP, 4, 0x00),
  RV32Ops.OR: lambda x, disp=0: enc_r(x, OPC_OP, 6, 0x00), RV32Ops.AND: lambda x, disp=0: enc_r(x, OPC_OP, 7, 0x00),
  # R-type M extension
  RV32Ops.MUL: lambda x, disp=0: enc_r(x, OPC_OP, 0, 0x01),
  RV32Ops.DIVU: lambda x, disp=0: enc_r(x, OPC_OP, 5, 0x01), RV32Ops.REMU: lambda x, disp=0: enc_r(x, OPC_OP, 7, 0x01),
  # I-type int. shift-immediates put shamt in the low 5 bits, SRAI additionally sets bit 30 (f7=0x20)
  RV32Ops.ADDI: lambda x, disp=0: enc_i(x, OPC_IMM, 0), RV32Ops.SLTIU: lambda x, disp=0: enc_i(x, OPC_IMM, 3),
  RV32Ops.XORI: lambda x, disp=0: enc_i(x, OPC_IMM, 4),
  RV32Ops.ORI: lambda x, disp=0: enc_i(x, OPC_IMM, 6), RV32Ops.ANDI: lambda x, disp=0: enc_i(x, OPC_IMM, 7),
  RV32Ops.SLLI: lambda x, disp=0: enc_shift_i(x, OPC_IMM, 1, 0x00), RV32Ops.SRLI: lambda x, disp=0: enc_shift_i(x, OPC_IMM, 5, 0x00),
  RV32Ops.SRAI: lambda x, disp=0: enc_shift_i(x, OPC_IMM, 5, 0x20),
  # loads / stores
  RV32Ops.LW: lambda x, disp=0: enc_i(x, OPC_LOAD, 2), RV32Ops.LBU: lambda x, disp=0: enc_i(x, OPC_LOAD, 4),
  RV32Ops.LHU: lambda x, disp=0: enc_i(x, OPC_LOAD, 5),
  RV32Ops.SB: lambda x, disp=0: enc_s(x, OPC_STORE, 0), RV32Ops.SH: lambda x, disp=0: enc_s(x, OPC_STORE, 1),
  RV32Ops.SW: lambda x, disp=0: enc_s(x, OPC_STORE, 2),
  # branches, +-4KiB, must be 2-byte aligned
  RV32Ops.BEQ: lambda x, disp=0: enc_b(x, OPC_BRANCH, 0, disp), RV32Ops.BNE: lambda x, disp=0: enc_b(x, OPC_BRANCH, 1, disp),
  RV32Ops.BLT: lambda x, disp=0: enc_b(x, OPC_BRANCH, 4, disp), RV32Ops.BGE: lambda x, disp=0: enc_b(x, OPC_BRANCH, 5, disp),
  RV32Ops.BLTU: lambda x, disp=0: enc_b(x, OPC_BRANCH, 6, disp), RV32Ops.BGEU: lambda x, disp=0: enc_b(x, OPC_BRANCH, 7, disp),
  # jumps. jal is +-1MiB pc-relative, jalr is a 12-bit offset off a register
  RV32Ops.JAL: lambda x, disp=0: enc_j(x, OPC_JAL, disp), RV32Ops.JALR: lambda x, disp=0: enc_i(x, OPC_JALR, 0),
  # upper immediate
  RV32Ops.LUI: lambda x, disp=0: enc_u(x, OPC_LUI), RV32Ops.AUIPC: lambda x, disp=0: enc_u(x, OPC_AUIPC),
  # system
  RV32Ops.CSRRS: lambda x, disp=0: enc_csr(x, OPC_SYSTEM, 2), RV32Ops.CSRRC: lambda x, disp=0: enc_csr(x, OPC_SYSTEM, 3),
  # takes no operands at all. fence iorw, iorw is what isa.py:75 always emitted
  RV32Ops.FENCE: lambda x, disp=0: struct.pack("<I", rv_i(OPC_FENCE, 0, 0, 0, 0x0FF)),
  # Tensix
  TensixOps.ADDDMAREG: lambda x, disp=0: enc_tt(x, 0x58, ((23, 23), (22, 12), (11, 6), (5, 0))),
  TensixOps.ATGETM: lambda x, disp=0: enc_tt(x, 0xA0, ((23, 0),)),
  TensixOps.ATRELM: lambda x, disp=0: enc_tt(x, 0xA1, ((23, 0),)),
  TensixOps.DMANOP: lambda x, disp=0: enc_tt(x, 0x60, ()),
  TensixOps.ELWADD: lambda x, disp=0: enc_tt(x, 0x28, ((23, 22), (21, 21), (20, 19), (18, 14), (13, 0))),
  TensixOps.ELWMUL: lambda x, disp=0: enc_tt(x, 0x27, ((23, 22), (21, 21), (20, 19), (18, 14), (13, 0))),
  TensixOps.INCRWC: lambda x, disp=0: enc_tt(x, 0x38, ((23, 18), (17, 14), (13, 10), (9, 6))),
  TensixOps.MOP: lambda x, disp=0: enc_tt(x, 0x01, ((23, 23), (22, 16), (15, 0))),
  TensixOps.MOVA2D: lambda x, disp=0: enc_tt(x, 0x12, ((23, 23), (22, 17), (16, 14), (13, 12), (11, 0))),
  TensixOps.MULDMAREG: lambda x, disp=0: enc_tt(x, 0x5A, ((23, 23), (22, 12), (11, 6), (5, 0))),
  TensixOps.MVMUL: lambda x, disp=0: enc_tt(x, 0x26, ((23, 22), (21, 19), (18, 14), (13, 0))),
  TensixOps.NOP: lambda x, disp=0: enc_tt(x, 0x02, ()),
  TensixOps.PACR: lambda x, disp=0: enc_tt(x, 0x41,
    ((23, 21), (20, 18), (17, 17), (16, 15), (14, 13), (12, 12), (11, 8), (7, 7), (6, 4), (3, 2), (1, 1), (0, 0))),
  TensixOps.RDCFG: lambda x, disp=0: enc_tt(x, 0xB1, ((23, 16), (15, 0))),
  TensixOps.REPLAY: lambda x, disp=0: enc_tt(x, 0x04, ((23, 14), (13, 4), (3, 1), (0, 0))),
  TensixOps.RMWCIB0: lambda x, disp=0: enc_tt(x, 0xB3, ((23, 16), (15, 8), (7, 0))),
  TensixOps.RMWCIB1: lambda x, disp=0: enc_tt(x, 0xB4, ((23, 16), (15, 8), (7, 0))),
  TensixOps.RMWCIB2: lambda x, disp=0: enc_tt(x, 0xB5, ((23, 16), (15, 8), (7, 0))),
  TensixOps.RMWCIB3: lambda x, disp=0: enc_tt(x, 0xB6, ((23, 16), (15, 8), (7, 0))),
  TensixOps.SEMGET: lambda x, disp=0: enc_tt(x, 0xA5, ((23, 2),)),
  TensixOps.SEMINIT: lambda x, disp=0: enc_tt(x, 0xA3, ((23, 20), (19, 16), (15, 2))),
  TensixOps.SEMPOST: lambda x, disp=0: enc_tt(x, 0xA4, ((23, 2),)),
  TensixOps.SEMWAIT: lambda x, disp=0: enc_tt(x, 0xA6, ((23, 15), (14, 2), (1, 0))),
  TensixOps.SETADC: lambda x, disp=0: enc_tt(x, 0x50, ((23, 21), (20, 20), (19, 18), (17, 0))),
  TensixOps.SETADCXX: lambda x, disp=0: enc_tt(x, 0x5E, ((23, 21), (20, 10), (9, 0))),
  TensixOps.SETADCXY: lambda x, disp=0: enc_tt(x, 0x51, ((23, 21), (20, 15), (14, 12), (11, 9), (8, 6), (5, 0))),
  TensixOps.SETADCZW: lambda x, disp=0: enc_tt(x, 0x54, ((23, 21), (20, 15), (14, 12), (11, 9), (8, 6), (5, 0))),
  TensixOps.SETC16: lambda x, disp=0: enc_tt(x, 0xB2, ((23, 16), (15, 0))),
  TensixOps.SETDMAREG: lambda x, disp=0: enc_tt(x, 0x45, ((23, 22), (21, 8), (7, 7), (6, 0))),
  TensixOps.SETRWC: lambda x, disp=0: enc_tt(x, 0x37, ((23, 22), (21, 18), (17, 14), (13, 10), (9, 6), (5, 0))),
  TensixOps.SFPADD: lambda x, disp=0: enc_tt(x, 0x85, ((23, 16), (15, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPADDI: lambda x, disp=0: enc_tt(x, 0x75, ((23, 8), (7, 4), (3, 0))),
  TensixOps.SFPARECIP: lambda x, disp=0: enc_tt(x, 0x99, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPCAST: lambda x, disp=0: enc_tt(x, 0x90, ((23, 8), (7, 4), (3, 0))),
  TensixOps.SFPCONFIG: lambda x, disp=0: enc_tt(x, 0x91, ((23, 8), (7, 4), (3, 0))),
  TensixOps.SFPENCC: lambda x, disp=0: enc_tt(x, 0x8A, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPEXEXP: lambda x, disp=0: enc_tt(x, 0x77, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPEXMAN: lambda x, disp=0: enc_tt(x, 0x78, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPIADD: lambda x, disp=0: enc_tt(x, 0x79, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPLOAD: lambda x, disp=0: enc_tt(x, 0x70, ((23, 20), (19, 16), (15, 13), (12, 0))),
  TensixOps.SFPLOADI: lambda x, disp=0: enc_tt(x, 0x71, ((23, 20), (19, 16), (15, 0))),
  TensixOps.SFPLUT: lambda x, disp=0: enc_tt(x, 0x73, ((23, 20), (19, 16), (15, 0))),
  TensixOps.SFPMAD: lambda x, disp=0: enc_tt(x, 0x84, ((23, 16), (15, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPMOV: lambda x, disp=0: enc_tt(x, 0x7C, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPMUL: lambda x, disp=0: enc_tt(x, 0x86, ((23, 16), (15, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPNOP: lambda x, disp=0: enc_tt(x, 0x8F, ()),
  TensixOps.SFPSETEXP: lambda x, disp=0: enc_tt(x, 0x82, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSHFT: lambda x, disp=0: enc_tt(x, 0x7A, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSHFT2: lambda x, disp=0: enc_tt(x, 0x94, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSTORE: lambda x, disp=0: enc_tt(x, 0x72, ((23, 20), (19, 16), (15, 13), (12, 0))),
  TensixOps.SFPSWAP: lambda x, disp=0: enc_tt(x, 0x92, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.STALLWAIT: lambda x, disp=0: enc_tt(x, 0xA2, ((23, 15), (14, 0))),
  TensixOps.STOREREG: lambda x, disp=0: enc_tt(x, 0x67, ((23, 18), (17, 0))),
  TensixOps.UNPACR: lambda x, disp=0: enc_tt(x, 0x42,
    ((23, 23), (22, 15), (14, 13), (12, 10), (9, 8), (7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1), (0, 0))),
  TensixOps.UNPACR_NOP: lambda x, disp=0: enc_tt(x, 0x43, ((23, 23), (22, 16), (15, 12), (11, 8), (7, 6), (5, 5), (4, 4), (3, 2), (1, 0))),
  TensixOps.WRCFG: lambda x, disp=0: enc_tt(x, 0xB0, ((23, 16), (15, 15), (14, 0))),
  TensixOps.ZEROACC: lambda x, disp=0: enc_tt(x, 0x10, ((23, 19), (18, 18), (17, 17), (16, 14), (13, 0))),
  TensixOps.ZEROSRC: lambda x, disp=0: enc_tt(x, 0x11, ((23, 4), (3, 3), (2, 2), (1, 0))),
  TensixOps.ADDRCRXY: lambda x, disp=0: enc_tt(x, 0x53, ((23, 21), (20, 15), (14, 12), (11, 9), (8, 6), (5, 0))),
  TensixOps.ADDRCRZW: lambda x, disp=0: enc_tt(x, 0x56, ((23, 21), (20, 15), (14, 12), (11, 9), (8, 6), (5, 0))),
  TensixOps.APOOL3S1: lambda x, disp=0: enc_tt(x, 0x25, ((23, 22), (21, 15), (14, 14), (13, 0))),
  TensixOps.APOOL3S2: lambda x, disp=0: enc_tt(x, 0x32, ((23, 22), (21, 15), (14, 14), (13, 0))),
  TensixOps.ATCAS: lambda x, disp=0: enc_tt(x, 0x64, ((23, 23), (22, 18), (17, 14), (13, 12), (11, 6), (5, 0))),
  TensixOps.ATINCGET: lambda x, disp=0: enc_tt(x, 0x61, ((23, 23), (22, 14), (13, 12), (11, 6), (5, 0))),
  TensixOps.ATINCGETPTR: lambda x, disp=0: enc_tt(x, 0x62, ((23, 23), (22, 22), (21, 18), (17, 14), (13, 12), (11, 6), (5, 0))),
  TensixOps.ATSWAP: lambda x, disp=0: enc_tt(x, 0x63, ((23, 23), (22, 14), (13, 6), (5, 0))),
  TensixOps.BITWOPDMAREG: lambda x, disp=0: enc_tt(x, 0x5B, ((23, 23), (22, 18), (17, 12), (11, 6), (5, 0))),
  TensixOps.CFGSHIFTMASK: lambda x, disp=0: enc_tt(x, 0xB8, ((23, 23), (22, 20), (19, 15), (14, 10), (9, 8), (7, 0))),
  TensixOps.CLEARDVALID: lambda x, disp=0: enc_tt(x, 0x36, ((23, 22), (21, 0))),
  TensixOps.CLREXPHIST: lambda x, disp=0: enc_tt(x, 0x21, ()),
  TensixOps.CMPDMAREG: lambda x, disp=0: enc_tt(x, 0x5D, ((23, 23), (22, 18), (17, 12), (11, 6), (5, 0))),
  TensixOps.CONV3S1: lambda x, disp=0: enc_tt(x, 0x22, ((23, 22), (21, 17), (16, 14), (13, 0))),
  TensixOps.CONV3S2: lambda x, disp=0: enc_tt(x, 0x23, ((23, 22), (21, 17), (16, 14), (13, 0))),
  TensixOps.DOTPV: lambda x, disp=0: enc_tt(x, 0x29, ((23, 22), (21, 21), (20, 19), (18, 14), (13, 0))),
  TensixOps.ELWSUB: lambda x, disp=0: enc_tt(x, 0x30, ((23, 22), (21, 21), (20, 19), (18, 14), (13, 0))),
  TensixOps.FLUSHDMA: lambda x, disp=0: enc_tt(x, 0x46, ((23, 0),)),
  TensixOps.GAPOOL: lambda x, disp=0: enc_tt(x, 0x34, ((23, 22), (21, 19), (18, 15), (14, 14), (13, 0))),
  TensixOps.GATESRCRST: lambda x, disp=0: enc_tt(x, 0x35, ((23, 1), (0, 0))),
  TensixOps.GMPOOL: lambda x, disp=0: enc_tt(x, 0x33, ((23, 22), (21, 19), (18, 15), (14, 14), (13, 0))),
  TensixOps.INCADCXY: lambda x, disp=0: enc_tt(x, 0x52, ((23, 21), (20, 15), (14, 12), (11, 9), (8, 6))),
  TensixOps.INCADCZW: lambda x, disp=0: enc_tt(x, 0x55, ((23, 21), (20, 15), (14, 12), (11, 9), (8, 6))),
  TensixOps.LOADIND: lambda x, disp=0: enc_tt(x, 0x49, ((23, 22), (21, 14), (13, 12), (11, 6), (5, 0))),
  TensixOps.LOADREG: lambda x, disp=0: enc_tt(x, 0x68, ((23, 18), (17, 0))),
  TensixOps.MFCONV3S1: lambda x, disp=0: enc_tt(x, 0x3A, ((23, 22), (21, 17), (16, 14), (13, 0))),
  TensixOps.MOP_CFG: lambda x, disp=0: enc_tt(x, 0x03, ((23, 0),)),
  TensixOps.MOVB2A: lambda x, disp=0: enc_tt(x, 0x0B, ((23, 17), (16, 14), (13, 12), (11, 0))),
  TensixOps.MOVB2D: lambda x, disp=0: enc_tt(x, 0x13, ((23, 23), (22, 17), (16, 14), (13, 11), (10, 0))),
  TensixOps.MOVD2A: lambda x, disp=0: enc_tt(x, 0x08, ((23, 23), (22, 17), (16, 14), (13, 12), (11, 0))),
  TensixOps.MOVD2B: lambda x, disp=0: enc_tt(x, 0x0A, ((23, 23), (22, 17), (16, 14), (13, 12), (11, 0))),
  TensixOps.MOVDBGA2D: lambda x, disp=0: enc_tt(x, 0x09, ((23, 23), (22, 17), (16, 14), (13, 12), (11, 0))),
  TensixOps.MOVDBGB2D: lambda x, disp=0: enc_tt(x, 0x0C, ((23, 23), (22, 17), (16, 14), (13, 11), (10, 0))),
  TensixOps.MPOOL3S1: lambda x, disp=0: enc_tt(x, 0x24, ((23, 22), (21, 15), (14, 14), (13, 0))),
  TensixOps.MPOOL3S2: lambda x, disp=0: enc_tt(x, 0x31, ((23, 22), (21, 15), (14, 14), (13, 0))),
  TensixOps.PACR_SETREG: lambda x, disp=0: enc_tt(x, 0x4A, ((23, 23), (22, 22), (21, 12), (11, 10), (9, 8), (7, 2), (1, 1), (0, 0))),
  TensixOps.RAREB: lambda x, disp=0: enc_tt(x, 0x15, ()),
  TensixOps.REG2FLOP: lambda x, disp=0: enc_tt(x, 0x48, ((23, 22), (21, 20), (19, 18), (17, 16), (15, 6), (5, 0))),
  TensixOps.RESOURCEDECL: lambda x, disp=0: enc_tt(x, 0x05, ((23, 13), (12, 4), (3, 0))),
  TensixOps.RSTDMA: lambda x, disp=0: enc_tt(x, 0x44, ()),
  TensixOps.SETASHRMH: lambda x, disp=0: enc_tt(x, 0x1E, ((23, 1), (0, 0))),
  TensixOps.SETASHRMH0: lambda x, disp=0: enc_tt(x, 0x1A, ((23, 1), (0, 0))),
  TensixOps.SETASHRMH1: lambda x, disp=0: enc_tt(x, 0x1B, ((23, 1), (0, 0))),
  TensixOps.SETASHRMV: lambda x, disp=0: enc_tt(x, 0x1C, ((23, 0),)),
  TensixOps.SETDVALID: lambda x, disp=0: enc_tt(x, 0x57, ((23, 0),)),
  TensixOps.SETIBRWC: lambda x, disp=0: enc_tt(x, 0x39, ((23, 18), (17, 6), (5, 0))),
  TensixOps.SETPKEDGOF: lambda x, disp=0: enc_tt(x, 0x1D, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPABS: lambda x, disp=0: enc_tt(x, 0x7D, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPAND: lambda x, disp=0: enc_tt(x, 0x7E, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPCOMPC: lambda x, disp=0: enc_tt(x, 0x8B, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPDIVP2: lambda x, disp=0: enc_tt(x, 0x76, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPGT: lambda x, disp=0: enc_tt(x, 0x97, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPLE: lambda x, disp=0: enc_tt(x, 0x96, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPLOADMACRO: lambda x, disp=0: enc_tt(x, 0x93, ((23, 20), (19, 16), (15, 13), (12, 0))),
  TensixOps.SFPLUTFP32: lambda x, disp=0: enc_tt(x, 0x95, ((23, 4), (3, 0))),
  TensixOps.SFPLZ: lambda x, disp=0: enc_tt(x, 0x81, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPMUL24: lambda x, disp=0: enc_tt(x, 0x98, ((23, 16), (15, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPMULI: lambda x, disp=0: enc_tt(x, 0x74, ((23, 8), (7, 4), (3, 0))),
  TensixOps.SFPNOT: lambda x, disp=0: enc_tt(x, 0x80, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPOR: lambda x, disp=0: enc_tt(x, 0x7F, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPPOPC: lambda x, disp=0: enc_tt(x, 0x88, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPPUSHC: lambda x, disp=0: enc_tt(x, 0x87, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSETCC: lambda x, disp=0: enc_tt(x, 0x7B, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSETMAN: lambda x, disp=0: enc_tt(x, 0x83, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSETSGN: lambda x, disp=0: enc_tt(x, 0x89, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPSTOCHRND: lambda x, disp=0: enc_tt(x, 0x8E, ((23, 21), (20, 16), (15, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPTRANSP: lambda x, disp=0: enc_tt(x, 0x8C, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SFPXOR: lambda x, disp=0: enc_tt(x, 0x8D, ((23, 12), (11, 8), (7, 4), (3, 0))),
  TensixOps.SHIFTDMAREG: lambda x, disp=0: enc_tt(x, 0x5C, ((23, 23), (22, 18), (17, 12), (11, 6), (5, 0))),
  TensixOps.SHIFTXA: lambda x, disp=0: enc_tt(x, 0x17, ((23, 2), (1, 0))),
  TensixOps.SHIFTXB: lambda x, disp=0: enc_tt(x, 0x18, ((23, 14), (13, 10), (9, 0))),
  TensixOps.STOREIND: lambda x, disp=0: enc_tt(x, 0x66, ((23, 23), (22, 22), (21, 21), (20, 14), (13, 12), (11, 6), (5, 0))),
  TensixOps.STREAMWAIT: lambda x, disp=0: enc_tt(x, 0xA7, ((23, 15), (14, 4), (3, 3), (2, 0))),
  TensixOps.STREAMWRCFG: lambda x, disp=0: enc_tt(x, 0xB7, ((23, 21), (20, 11), (10, 0))),
  TensixOps.SUBDMAREG: lambda x, disp=0: enc_tt(x, 0x59, ((23, 23), (22, 12), (11, 6), (5, 0))),
  TensixOps.TBUFCMD: lambda x, disp=0: enc_tt(x, 0x4B, ()),
  TensixOps.TRNSPSRCA: lambda x, disp=0: enc_tt(x, 0x14, ()),
  TensixOps.TRNSPSRCB: lambda x, disp=0: enc_tt(x, 0x16, ()),
  TensixOps.XMOV: lambda x, disp=0: enc_tt(x, 0x40, ((23, 23), (22, 0))),
}


# NOTE on branch fixups: x86.py:875 patches jump targets by splicing 4 bytes at a fixed offset, which works because
# x86 rel32 is a contiguous little-endian field. RV32 scatters the branch immediate across bits 31/30:25/11:8/7,
# so a fixup must *re-encode the whole word* rather than splice. Every RV32 instruction is exactly 4 bytes, so
# offsets are known before displacements are: lay out with disp=0, then call encodings[u.arg](u, disp=target-pc) again.
