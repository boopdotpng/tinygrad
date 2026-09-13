import os, struct, unittest
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Buffer, BufferSpec, BufferStorage
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
from tinygrad.engine.realize import run_linear, compile_linear, link_linear
from tinygrad.runtime.ops_tt import tt_program, program_records, DRAM
from tinygrad.runtime.support.tt.device import jal

# lui t0,3; addi t0,t0,0x280; lw t1,0(t0); lw t2,8(t0); sw t2,0(t1); return to resident BRISC.
WRITE_PARAM = struct.pack('<5I',0x32b7,0x28028293,0x0002a303,0x0082a383,0x00732023)+jal(0x34e0-(0x4000+20))

def raw_write(out:UOp, value:UOp):
  sink = UOp.sink(out, value, arg=KernelInfo('write_param'))
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg='raw RV32 parameter store'),
    UOp(Ops.BINARY,arg=tt_program({(1,2):{'brisc':WRITE_PARAM}}))))

class TestTTFormat(unittest.TestCase):
  def test_program_validation(self):
    self.assertEqual(program_records(tt_program({(1,2):{'brisc':WRITE_PARAM}}))[0][1]['brisc'],WRITE_PARAM)
    for lib in (b'bad',tt_program({(1,2):{'brisc':b'x'}}),tt_program({(1,2):{'unknown':b'1234'}})):
      with self.assertRaises(ValueError): program_records(lib)

@unittest.skipUnless(os.getenv('TT_TEST')=='1','set TT_TEST=1 for exclusive Blackhole hardware tests')
class TestTTHardware(unittest.TestCase):
  @classmethod
  def setUpClass(cls): cls.dev = Device[os.getenv('TT_TEST_DEVICE','TT')]

  def test_dma_views(self):
    for off,n in ((0,1),(1,3),(15,4099),(2047,8195),(65537,12345),(0,1<<20)):
      with self.subTest(offset=off,size=n):
        data = bytes(i%251 for i in range(n))
        a = Buffer('CPU',n,dtypes.uint8,initial_value=data)
        b = Buffer(self.dev.device,off+n+64,dtypes.uint8,preallocate=True)
        v = b.view(n,dtypes.uint8,off).ensure_allocated()
        v.copy_from(a)
        c = Buffer('CPU',n,dtypes.uint8,preallocate=True).copy_from(v)
        self.assertEqual(bytes(c.host[:]),data)

  def test_dram_to_dram(self):
    n = 16399
    a = Buffer('CPU',n,dtypes.uint8,initial_value=bytes(i%251 for i in range(n)))
    x,y = [Buffer(self.dev.device,n+32,dtypes.uint8,preallocate=True) for _ in range(2)]
    xv,yv = x.view(n,dtypes.uint8,1).ensure_allocated(), y.view(n,dtypes.uint8,17).ensure_allocated()
    xv.copy_from(a)
    yv.copy_from(xv)
    self.assertEqual(bytes(Buffer('CPU',n,dtypes.uint8,preallocate=True).copy_from(yv).host[:]),bytes(a.host[:]))

  def test_l1_dma(self):
    data = bytes(i%251 for i in range(8192))
    b = Buffer(self.dev.device,len(data),dtypes.uint8,opaque=BufferStorage(((1|2<<6)<<32)+0x90000))
    b.copy_from(Buffer('CPU',len(data),dtypes.uint8,initial_value=data))
    self.assertEqual(bytes(Buffer('CPU',len(data),dtypes.uint8,preallocate=True).copy_from(b).host[:]),data)

  def test_jit_dma_new_inputs(self):
    f = TinyJit(lambda x: x.to(self.dev.device).realize())
    for i in range(6): self.assertEqual(f(Tensor([i,i+1,i+2],device='CPU').realize()).tolist(),[i,i+1,i+2])

  def test_raw_dispatch(self):
    out = Buffer(self.dev.device,1,dtypes.uint32,opaque=BufferStorage(((1|2<<6)<<32)+0x90000))
    p = UOp.param(0,dtypes.uint32,1,device=self.dev.device)
    v = UOp.variable('value',0,100,dtypes.int,param=True)
    pv = v
    prg = raw_write(p,pv).replace(arg=ProgramInfo(globals=(0,),outs=(0,),vars=(pv,)))
    for value in (1,37,99):
      run_linear(UOp(Ops.LINEAR,src=(prg.call(UOp.from_buffer(out)),)),var_vals={'value':value})
      self.assertEqual(Buffer('CPU',1,dtypes.uint32,preallocate=True).copy_from(out).host.view(fmt='I')[0],value)

  def test_jit_raw_new_targets(self):
    image = WRITE_PARAM[:12] + struct.pack('<I',0x02a00393) + WRITE_PARAM[16:] # addi t2,zero,42
    def kernel(out):
      sink = UOp.sink(out,arg=KernelInfo('write_42'))
      return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg='raw RV32 store'),
        UOp(Ops.BINARY,arg=tt_program({(1,2):{'brisc':image}}))))
    f = TinyJit(lambda x: Tensor.custom_kernel(x,fxn=kernel)[0].realize())
    for i in range(6):
      t = Tensor.from_blob(((1|2<<6)<<32)+0x90000+i*64,(1,),dtype=dtypes.uint32,device=self.dev.device)
      self.assertEqual(f(t).tolist(),[42])

  def test_ring_wrap(self):
    h = Buffer(self.dev.device,64,dtypes.uint8,options=BufferSpec(host=True),initial_value=bytes(range(64)))
    d = Buffer(self.dev.device,64,dtypes.uint8,preallocate=True)
    x,y = UOp.from_buffer(d),UOp.from_buffer(h)
    linear = link_linear(compile_linear(UOp(Ops.LINEAR,src=(y.copy_to_device(self.dev.device).call(x,y),)),profile=False))
    for _ in range(65540): run_linear(linear,jit=True,update_stats=False)
    self.dev.synchronize()
    self.assertEqual(bytes(Buffer('CPU',64,dtypes.uint8,preallocate=True).copy_from(d).host[:]),bytes(range(64)))

  def test_allocation_reuse(self):
    self.dev.synchronize()
    spec = BufferSpec(host=True,nolru=True)
    def available(): return sum(b[0] for b in self.dev.sysmem.blocks.values() if b[3])
    before = available()
    for _ in range(10):
      buf = Buffer(self.dev.device,4096,dtypes.uint8,options=spec,preallocate=True)
      self.assertFalse(buf._buf & DRAM)
      buf.deallocate()
    self.assertEqual(before,available())

if __name__ == '__main__': unittest.main()
