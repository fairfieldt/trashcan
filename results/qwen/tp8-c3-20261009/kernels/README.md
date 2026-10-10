Experimental kernel sources of this round, as measured (none is in the engine). `qwen_q4cols_lds_body.glsl` is the
"columns outer" shared-memory design in its last form (opaque per-column masks, volatile weight view, clamped
tile loads, optional double buffering with `DOUBLE 1`); `qwen_q4cols_ldsx_body.glsl` the "rows outer" design
(tile loaded before the weights, per-row opaque tile offsets). `variants.md` lists the define-based variants of the
engine's `qwen_q4cols_body.glsl` and `matvecq4_body.glsl` and the `Rec::matrix` rule of the cluster binary. A
variant file is `#version 460` plus the defines plus `#include "<body>"`.
