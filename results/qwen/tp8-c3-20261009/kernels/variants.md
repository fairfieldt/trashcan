# Define-based variants of the engine's kernel bodies (round 3)

Variant files: `#version 460`, `#define COLS 3`, `#define ROWS R`, `#define WAVES W`, the defines below, then
`#include "qwen_q4cols_body.glsl"` (or `matvecq4_body.glsl` with `UNROLL`). Names in the microbenchmark files:
`qwen_q4c3_r5` (ROWS 5), `_r5w2` (ROWS 5, WAVES 2), `_r6`, `_k5120` (KFIX 5120), `_r5k5120`, `qwen_q4c3a` (ADDR),
`qwen_q4c3e` (EPI, also `_r5`, `_r2`), `qwen_q4c3p` (PREF, also `_r5`, `qwen_q4c2p`, `qwen_q4c4p`), `qwen_q4c3_dx0`,
`_dx1`, `_dnohi` (diagnostics, wrong results), `_p21`, `_p32`, `_dx1p16` (LDSPAD 21504, 32768, 16384),
`matvecq4_r3`, `_r3u2` (ROWS 3), `matvecq4_r2e`, `_r2u2e` (EPI), `matvecq4_r2p`, `_r4p` (PREF).

## `qwen_q4cols_body.glsl`

`KFIX` (row length compiled in): `#define KLEN KFIX` in place of `p.k` for `blocks=KLEN/32` and
`base[z]=min(t+z,nb-1)*(KLEN/4)`.

`ADDR` (row 0's indices once, the other rows one uniform add per load), in the row loop:
```glsl
uint sb=first*blocks+b,qb=p.stride+sb*4+part;
[[unroll]] for(uint r=0;r<ROWS;r++) {
    float scale=uintBitsToFloat(ww.words[sb+r*blocks]);
    uint qs=ww.words[qb+r*(blocks*4)];
    ...
```

`CLAMP_ROWS` (the five-row kernel on the cluster; phantom rows of a shard that is not a multiple of the
workgroup's rows read the last row, their results are not stored):
```glsl
uint block=min(first+r,p.n-1)*blocks+b;
```

`EPI` (all reductions before any store, one store block):
```glsl
float red[COLS][ROWS];
[[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++)red[z][r]=subgroupAdd(sums[z][r]);
if(lane==0)[[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++)
    if(first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=red[z][r];
```

`PREF` (the next iteration's weights loaded after row ROWS/2 through a volatile view, carried across the loop):
```glsl
layout(binding=0,std430) readonly volatile buffer WeightWordsV { uint words[]; } wwv;
...
float scale[ROWS];uint qs[ROWS];
{ uint b=min(lane/4,blocks-1); [[unroll]] for(uint r=0;r<ROWS;r++) { uint block=(first+r)*blocks+b;
  scale[r]=uintBitsToFloat(wwv.words[block]); qs[r]=wwv.words[p.stride+block*4+part]; } }
for(uint b=lane/4;b<blocks;b+=16) {
    ... x0, x1, bias as before ...
    float scale_n[ROWS];uint qs_n[ROWS];
    [[unroll]] for(uint r=0;r<ROWS;r++) {
        if(r==ROWS/2) {
            uint bn=min(b+16,blocks-1);
            [[unroll]] for(uint q=0;q<ROWS;q++) { uint block=(first+q)*blocks+bn;
                scale_n[q]=uintBitsToFloat(wwv.words[block]); qs_n[q]=wwv.words[p.stride+block*4+part]; }
        }
        vec4 lo=unpackUnorm4x8(qs[r]&0x0f0f0f0fu)*255.0;
        vec4 hi=unpackUnorm4x8((qs[r]>>4)&0x0f0f0f0fu)*255.0;
        [[unroll]] for(uint z=0;z<COLS;z++)sums[z][r]=fma(dot(lo,x0[z])+dot(hi,x1[z])-bias[z],scale[r],sums[z][r]);
    }
    [[unroll]] for(uint r=0;r<ROWS;r++){scale[r]=scale_n[r];qs[r]=qs_n[r];}
}
```

Diagnostics (wrong results, timing only): `DIAG_X0`: every column reads column 0's activations at offsets +z
(`x0[z]=b_v4.v[base[0]+b*8+part+z]`, same for `x1`); `DIAG_X1`: every column reads exactly column 0's activations
(the compiler then loads them once); `DIAG_NOHI`: `dot(hi,x1[z])` replaced by `hi.x`. `LDSPAD N`: `shared uint
pad[N/4];` kept alive by `if(p.r0==1u)pad[gl_LocalInvocationID.x]=lane; if(p.r1==1u)c_f.v[lane]=float(pad[gl_LocalInvocationID.x^1u]);`
(never taken), which caps the workgroups per CU; the compiler then also raises its VGPR allocation to that
occupancy.

## `matvecq4_body.glsl`

`EPI` and `PREF` as above with one column (`PREF` in the `UNROLL 1` loop only).

## `Rec::matrix` in the cluster binary (`engine/src/qwen/gpu.rs`)

```rust
3 if rows > 2048 && rows <= 2560 && c3_five_rows() => ("qwen_q4c3_r5", rows.div_ceil(20), 1),
3 => ("qwen_q4c3", rows.div_ceil(16), 1),
...
/// Whether three-column shards past 2048 rows take the five-row kernel (`TRASHCAN_QWEN_C3=r4`: never).
fn c3_five_rows() -> bool { std::env::var("TRASHCAN_QWEN_C3").map_or(true, |v| v != "r4") }
```
with `qwen_q4c3_r5` registered in `engine/build.rs`, `engine/src/vulkan.rs` and `micro::rows_per_workgroup` (20).
