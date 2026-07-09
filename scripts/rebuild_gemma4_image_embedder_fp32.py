#!/usr/bin/env python3
"""Rebuild the Gemma4 vision embedder Core ML package with fp32 compute.

The original fp16 conversion overflows (patch_dense output ~5e4, pos_norm
output ~750 squared in RMSNorm -> inf -> rsqrt -> 0), zeroing image_hidden.
Weights inside the int4 package are healthy, so we decompress them and
re-emit the same graph in fp32, then re-quantize the two big linears to int4.
Output is emitted as fp16 (values ~|32| max) so the Swift overlay keeps
binding image_hidden as Float16.
"""
import sys
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.optimize.coreml import decompress_weights
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

SRC = "runpod-artifacts/coreml-multimodal/gemma4_12b_image_embedder_patches32_int4_block32.mlpackage"
OUT_FP32 = sys.argv[1]
OUT_INT4 = sys.argv[2]

print("loading + decompressing", SRC, flush=True)
m = ct.models.MLModel(SRC, compute_units=ct.ComputeUnit.CPU_ONLY)
dm = decompress_weights(m)
f = dm._mil_program.functions["main"]

cval = lambda v: (np.asarray(v.val) if v is not None and v.val is not None else None)

ln_ops = [op for op in f.operations if op.op_type == "layer_norm"]
linear_ops = [op for op in f.operations if op.op_type == "linear"]
assert len(ln_ops) == 3 and len(linear_ops) == 2, (len(ln_ops), len(linear_ops))

def ln_params(op):
    return (
        cval(op.gamma).astype(np.float32),
        cval(op.beta).astype(np.float32),
        float(cval(op.epsilon)),
    )

g1, b1, eps1 = ln_params(ln_ops[0])   # patch_ln1 (6912)
g2, b2, eps2 = ln_params(ln_ops[1])   # patch_ln2 (3840)
g3, b3, eps3 = ln_params(ln_ops[2])   # pos_norm  (3840)

Wd = cval(linear_ops[0].weight).astype(np.float32)  # (3840, 6912)
bd = cval(linear_ops[0].bias)
bd = bd.astype(np.float32) if bd is not None else np.zeros(Wd.shape[0], np.float32)
Wp = cval(linear_ops[1].weight).astype(np.float32)  # (3840, 3840)

# positional embedding: the only rank-3 const with a size-2 axis and 3840 axis
pos_emb = None
for op in f.operations:
    if op.op_type == "const":
        a = cval(op.outputs[0])
        if a is not None and a.ndim == 3 and 3840 in a.shape and 2 in a.shape:
            pos_emb = a.astype(np.float32)
            pos_name = op.name
assert pos_emb is not None
print("pos_emb", pos_name, pos_emb.shape, flush=True)
# orient to (N, 2, 3840)
ax3840 = pos_emb.shape.index(3840)
pos_emb = np.moveaxis(pos_emb, ax3840, 2)
if pos_emb.shape[0] == 2 and pos_emb.shape[1] != 2:
    pos_emb = np.swapaxes(pos_emb, 0, 1)
N = pos_emb.shape[0]
Px = np.ascontiguousarray(pos_emb[:, 0, :])  # (N, 3840) x-axis table
Py = np.ascontiguousarray(pos_emb[:, 1, :])  # (N, 3840) y-axis table
print("pos tables", Px.shape, "N =", N, flush=True)

# RMSNorm eps: const feeding the add right after reduce_mean
rms_eps = None
for op in f.operations:
    if op.op_type == "add":
        xa, ya = cval(op.x), cval(op.y)
        for c in (xa, ya):
            if c is not None and c.size == 1 and 0 < float(c) < 1e-3:
                rms_eps = float(c)
assert rms_eps is not None
print("rms_eps", rms_eps, flush=True)

P, D, H = 32, 6912, 3840

@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, P, D), dtype=types.fp32),
        mb.TensorSpec(shape=(1, P, 2), dtype=types.int32),
    ],
    opset_version=ct.target.iOS18,
)
def vision(pixel_values, image_position_ids):
    h = mb.layer_norm(x=pixel_values, axes=[2], gamma=g1, beta=b1, epsilon=eps1)
    h = mb.linear(x=h, weight=Wd, bias=bd)
    h = mb.layer_norm(x=h, axes=[2], gamma=g2, beta=b2, epsilon=eps2)

    posf = mb.cast(x=image_position_ids, dtype="fp32")                    # (1,P,2)
    clamped = mb.clip(x=posf, alpha=0.0, beta=float(N - 1))
    idx = mb.cast(x=clamped, dtype="int32")
    x_idx = mb.slice_by_index(x=idx, begin=[0, 0, 0], end=[1, P, 1],
                              squeeze_mask=[False, False, True])          # (1,P)
    y_idx = mb.slice_by_index(x=idx, begin=[0, 0, 1], end=[1, P, 2],
                              squeeze_mask=[False, False, True])
    emb_x = mb.gather(x=Px, indices=x_idx, axis=0)                        # (1,P,H)
    emb_y = mb.gather(x=Py, indices=y_idx, axis=0)
    pos_x = mb.slice_by_index(x=posf, begin=[0, 0, 0], end=[1, P, 1])     # (1,P,1)
    pos_y = mb.slice_by_index(x=posf, begin=[0, 0, 1], end=[1, P, 2])
    valid_x = mb.cast(x=mb.not_equal(x=pos_x, y=-1.0), dtype="fp32")
    valid_y = mb.cast(x=mb.not_equal(x=pos_y, y=-1.0), dtype="fp32")
    pos_embs = mb.add(x=mb.mul(x=emb_x, y=valid_x), y=mb.mul(x=emb_y, y=valid_y))

    h = mb.add(x=h, y=pos_embs)
    h = mb.layer_norm(x=h, axes=[2], gamma=g3, beta=b3, epsilon=eps3)

    sq = mb.mul(x=h, y=h)
    ms = mb.reduce_mean(x=sq, axes=[2], keep_dims=True)
    r = mb.rsqrt(x=mb.add(x=ms, y=rms_eps))
    h = mb.mul(x=h, y=r)
    return mb.linear(x=h, weight=Wp, name="image_hidden")

print("converting fp32 mlprogram...", flush=True)
fixed = ct.convert(
    vision,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    # fp32 compute inside, but emit fp16 so the Swift overlay (which binds
    # image_hidden as Float16) keeps working. Values are ~|32| max, fp16-safe.
    outputs=[ct.TensorType(name="image_hidden", dtype=np.float16)],
)
fixed.save(OUT_FP32)
print("saved", OUT_FP32, flush=True)

# ---- validation helpers ----
def grid_pos():
    side = 6
    pos = np.zeros((1, P, 2), dtype=np.int32)
    for p in range(P):
        pos[0, p, 0] = p % side
        pos[0, p, 1] = p // side
    return pos

def solid(r, g, b):
    a = np.zeros((1, P, D), dtype=np.float32)
    a[0, :, :] = np.tile(np.array([r, g, b], np.float32), D // 3)
    return a

def np_reference(pix, pos):
    x = pix[0].astype(np.float64)
    def ln(v, g, b, e):
        mu = v.mean(-1, keepdims=True)
        var = v.var(-1, keepdims=True)
        return (v - mu) / np.sqrt(var + e) * g + b
    h = ln(x, g1, b1, eps1) @ Wd.T.astype(np.float64) + bd
    h = ln(h, g2, b2, eps2)
    pe = Px[pos[0, :, 0]] + Py[pos[0, :, 1]]
    h = ln(h + pe, g3, b3, eps3)
    h = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + rms_eps)
    return h @ Wp.T.astype(np.float64)

pos = grid_pos()
red = solid(0.72, -0.84, -0.84)
blue = solid(-0.84, -0.69, 0.72)

def run(model, pix):
    return np.asarray(list(model.predict(
        {"pixel_values": pix, "image_position_ids": pos}).values())[0])

hr, hb = run(fixed, red), run(fixed, blue)
ref = np_reference(red, pos)
print(f"[fp32] red maxAbs={np.abs(hr).max():.4g} nan={np.isnan(hr).sum()}"
      f" | blue maxAbs={np.abs(hb).max():.4g}"
      f" | red-vs-blue max|diff|={np.abs(hr-hb).max():.4g}"
      f" | vs-numpy max err={float(np.max(np.abs(hr-ref))):.4g}", flush=True)

print("quantizing int4 block32...", flush=True)
cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32))
q = cto.linear_quantize_weights(fixed, config=cfg)
q.save(OUT_INT4)
print("saved", OUT_INT4, flush=True)

hr, hb = run(q, red), run(q, blue)
print(f"[int4] red maxAbs={np.abs(hr).max():.4g} nan={np.isnan(hr).sum()}"
      f" | red-vs-blue max|diff|={np.abs(hr-hb).max():.4g}"
      f" | vs-numpy max err={float(np.max(np.abs(hr-ref))):.4g}", flush=True)
