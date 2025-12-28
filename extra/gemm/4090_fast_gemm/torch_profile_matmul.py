import torch
from torch.profiler import profile, ProfilerActivity, record_function

M = N = K = 8192
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
out = torch.empty((M, N), device="cuda", dtype=torch.float16)

# warmup
for _ in range(20):
  torch.matmul(a, b, out=out)
torch.cuda.synchronize()

with profile(
  activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
  record_shapes=True,
) as prof:
  with record_function("gemm"):
    for _ in range(10):
      torch.matmul(a, b, out=out)
  prof.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
prof.export_chrome_trace("trace.json")
