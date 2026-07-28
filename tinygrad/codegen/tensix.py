from dataclasses import replace
from typing import Callable

from tinygrad.helpers import ansistrip, getenv
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, graph_rewrite
from tinygrad.uop.render import pretty_print


# TT-specific scheduling rewrites land here. Keeping this matcher explicit lets the
# placeholder consume all of tinygrad's generic lowering while we add TT rules one
# at a time and move the hook earlier only when a rule needs unreduced structure.
tt_post_rewrite = PatternMatcher([])


def tt_to_sink(ast:UOp, renderer:Renderer, optimize:bool, default:Callable) -> UOp:
  """Lower a tinygrad kernel for TT, currently preserving the complete generic path."""
  if getenv("TT_DUMP_GRAPH", 1):
    print(f"\n--- TT input graph ---\n{ansistrip(pretty_print(ast))}\n--- end TT input graph ---")

  # Enter Scheduler for its generic range cleanup and KernelInfo construction, but
  # select no GPU opts. An explicit empty tuple takes precedence over both BEAM and
  # hand_coded_optimizations. TT chooses SFPU/FPU panels and core assignment itself.
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=()))
  sink = default(ast, renderer, optimize)
  return graph_rewrite(sink, tt_post_rewrite, ctx=renderer, name="TT post rewrite")
