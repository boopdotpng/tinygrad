import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, sint, AxisType
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

N = getenv("N", 4096)
M = K = N
run_count = getenv("CNT", 5)

def run_custom_uop():
  a = Tensor.uniform(N, N, dtype=dtypes.float16)
  b = Tensor.uniform(N, N, dtype=dtypes.float16)
  c = Tensor.empty(N, N, dtype=dtypes.float32) # this is probably default





  if getenv("VERIFY") == 1:
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a @ b).realize()
    with Context(DEBUG=0):
      err = (c - tc).square().mean().item()
    print(f"mean squared error{err}") 
    if err >= 1e-06: raise RuntimeError("matmul is wrong!")

# 197 Tflops on 5090, 94% of theoretical 209.5
import torch
@torch.no_grad()
def run_torch(iters: int = 2000, warmup: int = 50):
  torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  

  a = torch.randn((M, K), device="cuda", dtype=torch.float16)
  b = torch.randn((K, N), device="cuda", dtype=torch.float16)
  out = torch.empty((M, N), device="cuda", dtype=torch.float16)

  # Warm up (kernel selection, caches, lazy init).
  for _ in range(warmup):
    torch.mm(a, b, out=out)
  torch.cuda.synchronize()

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  start.record()
  for _ in range(iters):
    torch.mm(a, b, out=out)
  end.record()
  end.synchronize()  

  ms_per = start.elapsed_time(end) / iters
  sec_per = ms_per / 1e3

  flops = 2 * M * N * K  
  tflops = flops / sec_per / 1e12

  print(f"{tflops:.2f} Tflops in pytorch, {(tflops/209.5):.2%} of theoretical")


if __name__ == "__main__": 
  run_custom_uop() 
  if getenv("TORCH") == 1: run_torch()
