#include "common.glsl"
// COLS columns of a Q4_0 matrix product in the formulation of the one-column decode kernels (matvecq4,
// matvecq4narrow): each row's packed nibbles are converted once per block and shared by the columns, and
// each column keeps the decode kernels' lane-to-block mapping, zero-point bias, FMA chain and subgroup
// reduction, so every column matches a one-column decode bit for bit (the row layout per wave does not
// change a row's arithmetic).
// Weights in set 0 (p.stride: offset of the packed nibbles), x in b, y in c; p.n rows, p.k row length.
layout(local_size_x=WAVES*64) in;
void main() {
    uint lane=gl_SubgroupInvocationID,nb=batch(),t=gl_WorkGroupID.y*COLS;
    uint first=gl_WorkGroupID.x*(ROWS*WAVES)+gl_SubgroupID*ROWS,part=lane%4,blocks=p.k/32;
    if(t>=nb)return;
    float sums[COLS][ROWS];
    uint base[COLS];
    [[unroll]] for(uint z=0;z<COLS;z++){base[z]=min(t+z,nb-1)*(p.k/4);[[unroll]] for(uint r=0;r<ROWS;r++)sums[z][r]=0;}
    for(uint b=lane/4;b<blocks;b+=16) {
        vec4 x0[COLS],x1[COLS];float bias[COLS];
        [[unroll]] for(uint z=0;z<COLS;z++){
            x0[z]=b_v4.v[base[z]+b*8+part];x1[z]=b_v4.v[base[z]+b*8+part+4];
            bias[z]=8.0*dot(x0[z]+x1[z],vec4(1));
        }
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            uint block=(first+r)*blocks+b;
            float scale=uintBitsToFloat(ww.words[block]);
            uint qs=ww.words[p.stride+block*4+part];
            vec4 lo=unpackUnorm4x8(qs&0x0f0f0f0fu)*255.0;
            vec4 hi=unpackUnorm4x8((qs>>4)&0x0f0f0f0fu)*255.0;
            [[unroll]] for(uint z=0;z<COLS;z++)sums[z][r]=fma(dot(lo,x0[z])+dot(hi,x1[z])-bias[z],scale,sums[z][r]);
        }
    }
    [[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++) {
        float sum=subgroupAdd(sums[z][r]);
        if(lane==0&&first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=sum;
    }
}
