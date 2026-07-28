from dataclasses import replace
from typing import Callable

from tinygrad.device import Compiler
from tinygrad.helpers import ansistrip, getenv
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.tt.program import encode_tt_program
from tinygrad.uop.ops import KernelInfo, UOp


class TensixCompiler(Compiler):
  """Placeholder compiler producing a valid no-op five-stream TT program."""
  def __init__(self): super().__init__(None)
  def compile(self, src:str) -> bytes: return encode_tt_program({})


def format_tensix_uops(uops:list[UOp]) -> str:
  positions = {u:i for i,u in enumerate(uops)}
  lines = []
  for i,u in enumerate(uops):
    srcs = ", ".join(str(positions[s]) if s in positions else "ext" for s in u.src)
    arg = replace(u.arg, name=ansistrip(u.arg.name)) if isinstance(u.arg, KernelInfo) else u.arg
    lines.append(f"{i:04d}: {u.op.name:<12} dtype={u.dtype!s:<12} src=[{srcs}] arg={arg!r}")
  return "\n".join(lines)


class TensixRenderer(Renderer):
  """Placeholder TT renderer exposing tinygrad's lowered UOp stream."""
  device = "TT"
  has_local = False
  has_threads = False
  has_shared = False
  supports_float4 = False
  global_max = None
  local_max = None

  def __init__(self, target):
    super().__init__(target)
    self.compiler = TensixCompiler()

  def rewrite_to_sink(self, ast:UOp, optimize:bool, default:Callable) -> UOp:
    from tinygrad.codegen.tensix import tt_to_sink
    return tt_to_sink(ast, self, optimize, default)

  def render(self, uops:list[UOp]) -> str:
    source = format_tensix_uops(uops)
    if getenv("TT_DUMP_UOPS", 1): print(f"\n--- TT linearized UOps ({len(uops)}) ---\n{source}\n--- end TT linearized UOps ---")
    return source
