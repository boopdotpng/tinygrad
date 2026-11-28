import unittest, numpy as np
from tinygrad import Tensor, Device, TinyJit, Variable, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.tt import isa, ARG_BASE
from tinygrad.runtime.ops_tt import RING_DEPTH

ZERO, RA, T1, T2, T3, A0, A1, A2, A3, A4, A5, A6 = 0, 1, 6, 7, 28, 10, 11, 12, 13, 14, 15, 16
WORKER_ID, NOC_READ, NOC_WRITE, NOC_WAIT, SCRATCH = 0x28, 0x70, 0x74, 0x78, 0x100000 # worker L1: its id, the firmware's noc calls, free memory

def li(rd:int, v:int): return isa.addi(rd=rd, rs1=ZERO, imm=v)
def noc(fn:int, arg:int, nbytes:int) -> list: # read or write nbytes between the buffer of an arg and the scratch, and wait for it
  return [li(A0, 0), isa.lw(rd=A1, rs1=ZERO, imm=ARG_BASE + 8 * arg), isa.lw(rd=A2, rs1=ZERO, imm=ARG_BASE + 8 * arg + 4), li(A3, 0), li(A4, nbytes),
          isa.lui(rd=A5, imm=SCRATCH >> 12), li(A6, 0), isa.jalr(rd=RA, rs1=ZERO, imm=fn),
          li(A0, 0), li(A1, 0), li(A2, 1), isa.jalr(rd=RA, rs1=ZERO, imm=NOC_WAIT)]

def add_n(out:UOp, inp:UOp, n:UOp|None=None) -> UOp: # out = inp + n, or + 1 without n. every worker runs a program, the BRISC of worker 0 does it
  nbytes = out.numel() * 4
  body = [*noc(NOC_READ, 1, nbytes), isa.lui(rd=T1, imm=SCRATCH >> 12), isa.addi(rd=T2, rs1=T1, imm=nbytes),
          isa.lw(rd=A0, rs1=ZERO, imm=ARG_BASE + 16) if n is not None else li(A0, 1),
          isa.lw(rd=T3, rs1=T1, imm=0), isa.add(rd=T3, rs1=T3, rs2=A0), isa.sw(rs1=T1, rs2=T3, imm=0), isa.addi(rd=T1, rs1=T1, imm=4),
          isa.bne(rs1=T1, rs2=T2, imm=-16), *noc(NOC_WRITE, 0, nbytes)]
  insts = [isa.lw(rd=T1, rs1=ZERO, imm=WORKER_ID), isa.bne(rs1=T1, rs2=ZERO, imm=4 * (len(body) + 1)), *body]
  sink = UOp.sink(out.base, inp.base, *([n] if n is not None else []), arg=KernelInfo(f"add_n_{out.numel()}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp.native(x).call() for x in insts))))

def tt(x, dtype=np.uint32) -> Tensor: return Tensor(np.array(x, dtype=dtype), device=Device.DEFAULT).realize()
def empty(n:int) -> Tensor: return Tensor.empty(n, dtype=dtypes.uint32, device=Device.DEFAULT).contiguous().realize()
def add(x:Tensor, *n:Tensor) -> Tensor: return Tensor.custom_kernel(empty(x.numel()), x, *n, fxn=add_n)[0]

@unittest.skipUnless(Device.DEFAULT.split(":")[0] == "TT", "requires a TT device")
class TestTT(unittest.TestCase):
  def test_copy(self): # two buffers of a size share a staging buffer, so a copy that didn't happen reads the other's data
    for n in (1, 64, 4097, 1 << 20, (16 << 20) + 3):
      a, b = (np.random.randint(0, 255, n, dtype=np.uint8) for _ in range(2))
      ta, tb = tt(a, np.uint8), tt(b, np.uint8)
      for t, data in ((ta, a), (tb, b), (ta, a)): self.assertTrue(np.array_equal(t.numpy(), data))

  def test_copy_views(self): # an end that isn't 64 byte aligned goes through L1 a chunk at a time
    dram, host = Buffer(Device.DEFAULT, n:=1 << 18, preallocate=True), Buffer("CPU", n, preallocate=True)
    host.host[:] = bytes(np.random.randint(0, 255, n, dtype=np.uint8))
    dram.copy_from(host)
    for off, sz in ((0, 4096), (4, 100), (13, 4099), (64, 64), (1, 1), (4097, 30000), (3, 200000)):
      view, new = dram.view(sz, off).ensure_allocated(), Buffer("CPU", sz, initial_value=bytes((i * 7 + off) & 0xff for i in range(sz)))
      self.assertEqual(bytes(view.as_memoryview()), bytes(host.host[off:off+sz]))
      view.copy_from(new)
      host.host[off:off+sz] = new.host[:]
    self.assertEqual(bytes(dram.as_memoryview()), bytes(host.host[:]))

  def test_copy_dram_to_dram(self):
    a, data = Buffer(Device.DEFAULT, 1 << 16, preallocate=True), bytes(np.random.randint(0, 255, 1 << 16, dtype=np.uint8))
    b = Buffer(Device.DEFAULT, 1 << 16, initial_value=data)
    a.copy_from(b)
    self.assertEqual(bytes(a.as_memoryview()), data)
    a.view(1000, 77).ensure_allocated().copy_from(b.view(1000, 5).ensure_allocated())
    self.assertEqual(bytes(a.as_memoryview())[77:1077], data[5:1005])

  def test_many_host_buffers(self): # more than tt-kmd can pin at once
    for n in range(1, 40): self.assertEqual(tt(range(n * 1000)).tolist(), list(range(n * 1000)))

  def test_free_reuses_dram(self):
    alloc = Device[Device.DEFAULT].allocator
    for _ in range(4):
      free = [sum(len(v) for v in lv.values()) for lv in alloc.dram.storage]
      for _ in range(8): tt(np.arange(1 << 16)).numpy()
      alloc.free_cache()
      self.assertEqual([sum(len(v) for v in lv.values()) for lv in alloc.dram.storage], free)

  def test_program(self): self.assertEqual(add(tt(range(16))).tolist(), list(range(1, 17)))

  def test_program_variable(self):
    a = tt(range(16))
    for i in (0, 1, 100):
      self.assertEqual(add(a, Tensor(Variable("n", 0, 1000, dtypes.int).bind(i))).tolist(), [x + i for x in range(16)])

  def test_program_chain(self): # one batch: the copy in, three launches that depend on each other, the copy out
    x = Tensor(np.arange(32, dtype=np.uint32), device=Device.DEFAULT)
    self.assertEqual(add(add(add(x))).tolist(), list(range(3, 35)))

  def test_program_cache(self): # more programs than the L1 of a worker tracks, each run twice
    for _ in range(2):
      for n in range(4, 300, 4): self.assertEqual(add(tt(range(n))).tolist(), list(range(1, n + 1)))

  def test_jit(self):
    @TinyJit
    def f(a): return add(add(a)).realize()
    for i in range(6): self.assertEqual(f(tt(range(i, i + 16))).tolist(), list(range(i + 2, i + 18)))

  def test_jit_variable(self):
    @TinyJit
    def f(a, n): return add(a, Tensor(n)).realize()
    a = tt(range(16))
    for i in range(1, 6): self.assertEqual(f(a, Variable("n", 0, 1000, dtypes.int).bind(i)).tolist(), list(range(i, i + 16)))

  def test_ring_wraps(self): # a queue is RING_DEPTH entries: submit past it without synchronizing
    a = tt(range(16))
    for _ in range(RING_DEPTH * 3): a = add(a).realize()
    self.assertEqual(a.tolist(), list(range(RING_DEPTH * 3, RING_DEPTH * 3 + 16)))

  def test_ring_full(self): # the dram copies take longer than their submits: the host has to wait for a free ring entry
    a, b = (Buffer(Device.DEFAULT, n:=64 << 20, preallocate=True) for _ in range(2))
    host = Buffer("CPU", n, initial_value=np.random.randint(0, 255, n, dtype=np.uint8).data)
    a.copy_from(host)
    for i in range(RING_DEPTH * 2): (a, b)[(i + 1) % 2].copy_from((a, b)[i % 2])
    self.assertEqual(bytes(a.as_memoryview()), bytes(host.host[:]))

  def test_lowering_not_implemented(self):
    with self.assertRaises(NotImplementedError): (tt(range(16)) + 1).realize()

if __name__ == "__main__":
  unittest.main()
