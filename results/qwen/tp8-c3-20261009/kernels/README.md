Experimental kernel sources of this round, as measured (none is in the engine). `qwen_q4cols_lds_body.glsl` is the
"columns outer" shared-memory design in its last form (opaque per-column masks, volatile weight view, clamped
tile loads, optional double buffering with `DOUBLE 1`); `qwen_q4cols_ldsx_body.glsl` the "rows outer" design
(tile loaded before the weights, per-row opaque tile offsets). `variants.md` lists the define-based variants of the
engine's `qwen_q4cols_body.glsl` and `matvecq4_body.glsl` and the `Rec::matrix` rule of the cluster binary. A
variant file is `#version 460` plus the defines plus `#include "<body>"`.

`tiled-layout-and-bias.patch` (round 3b): the tiled Q4_0 repack (`quant::repack_q4_tiled`, the `q4t_scale`/`q4t_qs`
helpers of common.glsl), the `TILED` loops of `qwen_q4cols_body.glsl` and `matvecq4_body.glsl` (uniform tile
bases, constant row offsets for full tiles, a partial last tile), the `BIAS` variant reading precomputed
zero-point biases from buffer d, the variant files (`qwen_q4c3t`, `qwen_q4c2t`, `qwen_q4c3b`, `qwen_q4c3bt`,
`matvecq4_r2t`, `matvecq4_r2u2t`) and the `qwen-matvec-bench` support (tiled copies of the matrices for kernels
named `*t`, the bias buffer for all). `git apply` on eabff7b..ffb3076 restores it.
