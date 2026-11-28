import struct
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.helpers import panic
from tinygrad.renderer.x86 import ISARenderer
from tinygrad.renderer.tt import isa

# a program is the code of the five RISCs of a Tensix core, each runs its own with the kernel args at ARG_BASE in L1
ROLES = ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2")
ARG_BASE, MAX_ARGS = 0x80, 16 # an arg is 8 bytes
PROG_HDR, MAX_CODE = 64, 0x1ac00 # the L1 a worker caches programs in
RET = struct.pack("<2I", isa.lw(rd=5, rs1=3, imm=-16), isa.jalr(rd=0, rs1=5)) # back to the worker firmware: lw t0, -16(gp); jr t0

def pack_program(code:dict[str, bytes], cb_mask:int=0xffffffff) -> bytes:
  # [the offset of each role's code, 0 if it has none][the circular buffers to reset before a launch], then the code
  assert set(code) <= set(ROLES), f"unknown roles {set(code) - set(ROLES)}"
  body, offs = bytearray(), []
  for role in ROLES:
    body += bytes(-len(body) % 16)
    offs.append(PROG_HDR + len(body) if role in code else 0)
    if role in code: body += code[role] + RET
  body += bytes(-len(body) % 64)
  if len(body) > MAX_CODE: raise RuntimeError(f"program of {len(body)} bytes does not fit the {MAX_CODE} bytes of L1 a worker caches programs in")
  return struct.pack(f"<6I{PROG_HDR-24}x", *offs, cb_mask) + bytes(body)

class TTRenderer(ISARenderer):
  device = "TT"
  # TODO: lower uops to the RV32 and Tensix instructions of the five RISCs
  pre_isel_matcher = PatternMatcher([(UPat(Ops.SINK), lambda: panic(NotImplementedError, "TT can't lower kernels yet, only run assembled programs"))])
  isel_matcher = pre_regalloc_matcher = post_regalloc_matcher = PatternMatcher([])

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # the instructions are isa words, a str switches the role the next ones are for
    code:dict[str, list[int]] = {}
    role = ROLES[0]
    for ins in (u.body.arg.code for u in lin.src):
      if isinstance(ins, str): role = ins
      else: code.setdefault(role, []).append(isa.inline(ins) if isinstance(ins, isa.Tensix) else int(ins))
    return pack_program({r: struct.pack(f"<{len(words)}I", *words) for r, words in code.items()})
