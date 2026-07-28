from tinygrad.renderer import Renderer


class TensixRenderer(Renderer):
  """Tensix launches one cooperating five-RISC program per worker core, not GPU thread groups."""
  has_local = False
  has_threads = False
