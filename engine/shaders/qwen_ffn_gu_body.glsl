#include "common.glsl"
// Gate and up projections of a SwiGLU FFN and its activation in one dispatch (docs/tp8-optimization.md F1). The gate
// matrix is the weight binding, the up matrix (same shape and repacked layout) binding d. Each column keeps the
// decode kernels' lane-to-block mapping, zero-point bias, FMA chain and subgroup reduction for both matrices
// (matvecq4 for one column, qwen_q4cN for more) and the output is qwen_swiglu's silu(g) * u, so it equals those
// three dispatches bit for bit. Written for any subgroup size S: a subgroup takes ROWS rows and S/4 blocks per step
// (16 rows per workgroup with S = 64).
// b: x [t][p.k], c: activation [t][p.n], d: up matrix. p.n rows, p.k row length, p.stride: word offset of the
// packed nibbles in both matrices.
#define ROWS 4
layout(local_size_x=256) in;
void main() {
    uint lane=gl_SubgroupInvocationID,nb=batch(),t=gl_WorkGroupID.y*COLS,S=gl_SubgroupSize;
    uint first=gl_WorkGroupID.x*(ROWS*gl_NumSubgroups)+gl_SubgroupID*ROWS,part=lane%4,blocks=p.k/32;
    if(t>=nb)return;
    float sg[COLS][ROWS],su[COLS][ROWS];
    uint base[COLS];
    [[unroll]] for(uint z=0;z<COLS;z++){base[z]=min(t+z,nb-1)*(p.k/4);[[unroll]] for(uint r=0;r<ROWS;r++){sg[z][r]=0;su[z][r]=0;}}
    for(uint b=lane/4;b<blocks;b+=S/4) {
        vec4 x0[COLS],x1[COLS];float bias[COLS];
        [[unroll]] for(uint z=0;z<COLS;z++){
            x0[z]=b_v4.v[base[z]+b*8+part];x1[z]=b_v4.v[base[z]+b*8+part+4];
            bias[z]=8.0*dot(x0[z]+x1[z],vec4(1));
        }
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            uint block=(first+r)*blocks+b;
            float gs=uintBitsToFloat(ww.words[block]),us=uintBitsToFloat(d_u.v[block]);
            uint gq=ww.words[p.stride+block*4+part],uq=d_u.v[p.stride+block*4+part];
            vec4 glo=unpackUnorm4x8(gq&0x0f0f0f0fu)*255.0;
            vec4 ghi=unpackUnorm4x8((gq>>4)&0x0f0f0f0fu)*255.0;
            vec4 ulo=unpackUnorm4x8(uq&0x0f0f0f0fu)*255.0;
            vec4 uhi=unpackUnorm4x8((uq>>4)&0x0f0f0f0fu)*255.0;
            [[unroll]] for(uint z=0;z<COLS;z++){
                sg[z][r]=fma(dot(glo,x0[z])+dot(ghi,x1[z])-bias[z],gs,sg[z][r]);
                su[z][r]=fma(dot(ulo,x0[z])+dot(uhi,x1[z])-bias[z],us,su[z][r]);
            }
        }
    }
    [[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++) {
        float g=subgroupAdd(sg[z][r]),u=subgroupAdd(su[z][r]);
        if(lane==0&&first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=g/(1.0+exp(-g))*u;
    }
}
