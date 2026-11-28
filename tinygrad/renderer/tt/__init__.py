import struct
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, AxisType, GroupOp, graph_rewrite, identity_element
from tinygrad.uop.symbolic import sym, pm_move_where_on_load
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_unparented
from tinygrad.helpers import panic, prod, round_up, ceildiv
from tinygrad.renderer.x86 import ISARenderer
from tinygrad.renderer.tt import isa

# a program is the code of the five RISCs of a Tensix core, each runs its own with the kernel args at ARG_BASE in L1
ROLES = ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2")
ARG_BASE, MAX_ARGS = 0x80, 16 # an arg is 8 bytes
PROG_HDR, MAX_CODE = 64, 0x1ac00 # the L1 a worker caches programs in
RET = struct.pack("<2I", isa.lw(rd=5, rs1=3, imm=-16), isa.jalr(rd=0, rs1=5)) # back to the worker firmware: lw t0, -16(gp); jr t0

# tile geometry: MVMUL operates on (8x16) x (16x16) -> (8x16) faces; 1D elementwise packs 128 elements per face
TILE_M, TILE_N, TILE_K, FACE_ELEMS = 8, 16, 16, 128
TILE_ELEMS = 32 * 32
PACKED_TILE_BYTES = {"bfp8": 1088, "bfp4": 576} # 1024-elem tile: 64B shared exponents + mantissas

def tile_nbytes(dtype:DType|str) -> int:
  if isinstance(dtype, str): return PACKED_TILE_BYTES[dtype]
  return PACKED_TILE_BYTES.get(dtype.name, TILE_ELEMS * dtype.itemsize)

def face_nbytes(dtype:DType|str, face_rows:int=TILE_M) -> int:
  # packed block-float faces share one 16B exponent header across 16x16 elements (4 faces per 32x32 tile)
  if isinstance(dtype, str) or dtype.name in PACKED_TILE_BYTES: return tile_nbytes(dtype) // (TILE_ELEMS // (face_rows * TILE_N))
  return face_rows * TILE_N * dtype.itemsize

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

def _pad_and_upcast(k:Scheduler, rng:UOp, q:int) -> tuple[UOp, UOp]:
  # coarse pad + split into (outer, inner=UPCAST(q)); allows padding small dims (e.g. 4 -> 128) without GPU PADTO's 4x cap
  if rng.src[0].op is Ops.CONST and (sz:=int(rng.vmax + 1)) % q != 0:
    new_sz = round_up(sz, q)
    new_rng = rng.replace(src=(rng.src[0].const_like(new_sz),))
    valid = new_rng < sz
    gate = lambda b: b.replace(src=(b.src[0], b.src[1].get_idx().valid(valid & b.src[1].get_valid())))
    replaces: dict[UOp, UOp] = {rng: new_rng}
    for b in k.bufs:
      if rng in b.src[1].ranges: replaces[b] = gate(b)
    for u in k.ast.backward_slice:
      if u.op is Ops.STORE and u.src[0].op is Ops.INDEX and rng in u.ranges: replaces[u] = u.replace(src=(gate(u.src[0]),) + u.src[1:])
    for r in k.reduceops:
      if rng in r.src[1:]: replaces[r] = r.replace(src=(valid.where(r.src[0], UOp.const(identity_element(r.arg[0], r.dtype), r.dtype)),) + r.src[1:])
    k.ast = k.ast.substitute(replaces, f"tt_padto {rng.axis_id} {q}")
    rng = new_rng
  return k.shift_to(rng, q, AxisType.UPCAST)

def _match_mvmul(r:UOp) -> tuple[UOp, UOp, list[UOp], list[UOp], list[UOp]]|None:
  if r.arg[0] is not Ops.ADD: return None
  body = r.src[0]
  if body.op is Ops.WHERE: body = body.src[1]
  if body.op is Ops.CAST: body = body.src[0]
  if body.op is not Ops.MUL: return None
  a, b = body.src
  rk = sorted([u for u in r.src[1:] if u in a.ranges and u in b.ranges], key=lambda x: x.arg)
  rm = sorted([u for u in a.ranges if u not in b.ranges and u not in r.src[1:]], key=lambda x: x.arg)
  rn = sorted([u for u in b.ranges if u not in a.ranges and u not in r.src[1:]], key=lambda x: x.arg)
  return (a, b, rm, rn, rk) if rk and (rm or rn) else None

def _tt_tile(sink:UOp, ren:"TTRenderer") -> UOp:
  k = Scheduler(sink, ren)
  k.convert_weak_to_global()

  # 1. FPU contractions (MVMUL): 2D face rule (M=8, N=16, K=16), supports M=1 GEMV and multiple contractions (e.g. SwiGLU)
  upcasted: dict[UOp, tuple[UOp, UOp]] = {}
  def ensure_upcast(rng:UOp, q:int) -> tuple[UOp, UOp]:
    if rng not in upcasted: upcasted[rng] = _pad_and_upcast(k, rng, q)
    return upcasted[rng]

  mvmuls = [(r, m) for r in k.reduceops if (m:=_match_mvmul(r)) is not None]
  if mvmuls:
    ik_by_red: list[UOp] = []
    for _, (_, _, rm, rn, rk) in mvmuls:
      if rm: ensure_upcast(rm[-1], TILE_M)
      if rn: ensure_upcast(rn[-1], TILE_N)
      ik_by_red.append(ensure_upcast(rk[-1], TILE_K)[1])
    for ik in ik_by_red:
      r_now = next(u for u in k.reduceops if ik in u.src[1:])
      r0 = r_now.src[0]
      gate, mul = (r0.src[0], r0.src[1]) if r0.op is Ops.WHERE else (None, r0)
      if mul.op is Ops.CAST: mul = mul.src[0]
      ins = mul.src if gate is None else tuple(gate.where(x, UOp.const(0, x.dtype)) for x in mul.src)
      wmma = UOp.wmma(ins[0], ins[1], UOp.const(0.0, r_now.dtype), (TILE_N, TILE_M, TILE_K), 1)
      outer_reds = tuple(u for u in r_now.src[1:] if u is not ik)
      k.ast = k.ast.substitute({r_now: UOp(Ops.REDUCE, src=(wmma,) + outer_reds, arg=(Ops.ADD, 0)) if outer_reds else wmma})
  else:
    out_rngs = [k.rngs[i] for i in k.axes_of(AxisType.GLOBAL)]
    red_rngs = [k.rngs[i] for i in k.axes_of(AxisType.WEAK, reduce=True)]
    if red_rngs:
      # 2. Horizontal / row reductions: reduce axis by 16, output row axis (if any) by 8
      ensure_upcast(red_rngs[-1], TILE_K)
      if out_rngs: ensure_upcast(out_rngs[-1], TILE_M)
    elif len(out_rngs) == 1:
      # 3a. 1D elementwise (non-MVMUL): N % 128 == 0
      ensure_upcast(out_rngs[0], FACE_ELEMS)
    elif len(out_rngs) >= 2:
      # 3b. Unmerged multi-axis elementwise (e.g. 2D broadcast / RoPE): (8, 16) face
      ensure_upcast(out_rngs[-1], TILE_N)
      ensure_upcast(out_rngs[-2], TILE_M)

  return graph_rewrite(k.ast, sym + pm_move_where_on_load + pm_flatten_range + pm_reduce_unparented, name="tt tile simplify")

def _apply_sharding(sink:UOp, gset:set[UOp], rng_subs:dict[UOp, UOp], added_loops:tuple[UOp, ...], valid:UOp|None) -> UOp:
  end_gate: dict[UOp, UOp] = {}
  for u in sink.toposort():
    if u.op is Ops.END:
      kept = tuple(r for r in u.src[1:] if r not in gset) + added_loops
      end_gate[u] = u.replace(src=(u.src[0],) + kept) if kept else u.src[0]
    elif valid is not None and u.op is Ops.INDEX and any(r in gset for r in u.src[1].ranges):
      end_gate[u] = u.replace(src=(u.src[0], u.src[1].get_idx().valid(valid & u.src[1].get_valid())))
    elif valid is not None and u.op is Ops.STORE and u.src[0].op is Ops.INDEX and any(r in gset for r in u.ranges):
      idx = u.src[0]
      end_gate[u] = u.replace(src=(idx.replace(src=(idx.src[0], idx.src[1].get_idx().valid(valid & idx.src[1].get_valid()))),) + u.src[1:])
  sink = sink.substitute(end_gate).substitute(rng_subs, walk=True)
  return graph_rewrite(sink, sym + pm_move_where_on_load + pm_flatten_range, name="tt shard cores")

def _tt_shard_cores(sink:UOp, cols:int) -> UOp:
  global_rngs = sorted([u for u in sink.toposort() if u.op is Ops.RANGE and u.axis_type is AxisType.GLOBAL], key=lambda x: x.arg)
  if not global_rngs or any(g.src[0].op is not Ops.CONST for g in global_rngs): return sink
  next_slot = max((u.axis_id[0] for u in sink.toposort() if u.op is Ops.RANGE), default=-1) + 1

  # 2D (gx, gy) grid for WMMAs that have both M-only (mf >= 2) and N-only (nf >= 2) GLOBAL axes -> enables row/col multicast
  wmmas = [u for u in sink.toposort() if u.op is Ops.WMMA]
  if wmmas:
    a, b = wmmas[0].src[:2]
    gms = [g for g in global_rngs if g in a.ranges and g not in b.ranges and int(g.vmax + 1) >= 2]
    gns = [g for g in global_rngs if g in b.ranges and g not in a.ranges and int(g.vmax + 1) >= 2]
    if gms and gns:
      gm, gn = gms[-1], gns[-1]
      mf, nf = int(gm.vmax + 1), int(gn.vmax + 1)
      chunk_n = ceildiv(nf, min(nf, cols - 1)) # cols - 1 full 10-row columns (avoids CP/DMA cutout)
      gx = ceildiv(nf, chunk_n)
      chunk_m = ceildiv(mf, min(mf, 10))
      gy = ceildiv(mf, chunk_m)
      if gx > 1 and gy > 1:
        outer_n = UOp.range(chunk_n, next_slot, AxisType.LOOP, dtype=gn.dtype) if chunk_n > 1 else None
        outer_m = UOp.range(chunk_m, next_slot + (outer_n is not None), AxisType.LOOP, dtype=gm.dtype) if chunk_m > 1 else None
        coord_n = UOp.special(gx, "gidx0") * chunk_n + outer_n if outer_n is not None else UOp.special(gx, "gidx0")
        coord_m = UOp.special(gy, "gidx1") * chunk_m + outer_m if outer_m is not None else UOp.special(gy, "gidx1")
        valids = ([coord_n < nf] if gx * chunk_n > nf else []) + ([coord_m < mf] if gy * chunk_m > mf else [])
        valid = valids[0] & valids[1] if len(valids) == 2 else (valids[0] if valids else None)
        batch_loops = {g: g.replace(arg=(AxisType.LOOP,) + g.axis_id) for g in global_rngs if g not in (gm, gn)}
        added = tuple(batch_loops.values()) + tuple(r for r in (outer_m, outer_n) if r is not None)
        return _apply_sharding(sink, set(global_rngs), {gn: coord_n, gm: coord_m} | batch_loops, added, valid)

  # 1D contiguous sharding across up to (cols * 10 - 2) Tensix workers (enables CB-page NoC coalescing)
  dims = [int(g.vmax + 1) for g in global_rngs]
  total = prod(dims)
  has_compute = bool(wmmas) or any(u.op is Ops.REDUCE for u in sink.toposort())
  min_faces = 1 if has_compute else 8 # 1 full 32x32 tile (8 faces = 1024 elems) per core for pure elementwise
  cores = min(cols * 10 - 2, max(1, ceildiv(total, min_faces)))
  chunk = ceildiv(total, cores)
  cores = ceildiv(total, chunk) # trim idle trailing cores when chunking leaves them with zero work
  if cores == 1 and total == 1: return sink

  core_id = UOp.special(cores, "gidx0") if cores > 1 else UOp.const(0, global_rngs[0].dtype)
  outer = UOp.range(chunk, next_slot, AxisType.LOOP, dtype=global_rngs[0].dtype) if chunk > 1 else None
  flat = core_id * chunk + outer if outer is not None else core_id
  valid = flat < total if cores * chunk > total else None

  rng_subs: dict[UOp, UOp] = {}
  stride = total
  for g, d in zip(global_rngs, dims):
    stride //= d
    coord = flat // stride if stride > 1 else flat
    if d < total // stride or stride < total: coord = coord % d
    rng_subs[g] = coord
  return _apply_sharding(sink, set(global_rngs), rng_subs, (outer,) if outer is not None else (), valid)

class TTRenderer(ISARenderer):
  device = "TT"
  has_local = True
  has_shared = False
  # keep transcendentals and SFPU/FPU ops undecomposed; RV32 index math still gets integer decomps later
  code_for_op = {op: lambda *_: "" for op in (*GroupOp.ALU, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.RECIP, Ops.SIN)}
  # TODO: lower tiled uops to space transfers and RV32/Tensix instructions of the five RISCs
  pre_isel_matcher = PatternMatcher([(UPat(Ops.SINK), lambda: panic(NotImplementedError, "TT can't lower kernels yet, only run assembled programs"))])
  isel_matcher = pre_regalloc_matcher = post_regalloc_matcher = PatternMatcher([])

  @property
  def cols(self) -> int: return self.target.get("cols", 14)
  @property
  def max_cores(self) -> int: return self.cols * 10 - 2

  def supported_dtypes(self) -> set[DType]: return {dtypes.float32, dtypes.bfloat16, dtypes.int32, dtypes.uint32, dtypes.bool}

  def rewrite_to_sink(self, ast:UOp, optimize:bool=True) -> UOp:
    if not optimize: return ast
    return _tt_shard_cores(_tt_tile(ast, self), self.cols)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # the instructions are isa words, a str switches the role the next ones are for
    code:dict[str, list[int]] = {}
    role = ROLES[0]
    for ins in (u.body.arg.code for u in lin.src):
      if isinstance(ins, str): role = ins
      else: code.setdefault(role, []).append(isa.inline(ins) if isinstance(ins, isa.Tensix) else int(ins))
    return pack_program({r: struct.pack(f"<{len(words)}I", *words) for r, words in code.items()})
