#include "common.glsl"
// matvecq4 with ROWS rows per wave and WAVES waves per workgroup (matvecq4 itself: 4 and 4). Every row keeps
// matvecq4's lane-to-block mapping, zero-point bias, FMA chain and subgroup reduction, so a row's result is the
// same bits whatever the geometry; the geometry only changes how many waves a shard dispatches and how often
// the activation vector is re-read. With UNROLL 2 each lane issues the loads of two blocks (b, b + 16) before
// using them, and still accumulates them in that order.
layout(local_size_x=WAVES*64) in;
void main() {
    uint lane=gl_SubgroupInvocationID;
    uint first=gl_WorkGroupID.x*(ROWS*WAVES)+gl_SubgroupID*ROWS, part=lane%4, blocks=p.k/32;
    float sums[ROWS];
    [[unroll]] for(uint r=0;r<ROWS;r++) sums[r]=0;
    uint b=lane/4;
#if UNROLL==2
    for(;b+16<blocks;b+=32) {
        vec4 x0=b_v4.v[b*8+part],x1=b_v4.v[b*8+part+4];
        vec4 y0=b_v4.v[(b+16)*8+part],y1=b_v4.v[(b+16)*8+part+4];
        float scale[ROWS],scale2[ROWS];uint qs[ROWS],qs2[ROWS];
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            uint block=(first+r)*blocks+b;
            scale[r]=uintBitsToFloat(ww.words[block]);
            qs[r]=ww.words[p.stride+block*4+part];
            scale2[r]=uintBitsToFloat(ww.words[block+16]);
            qs2[r]=ww.words[p.stride+(block+16)*4+part];
        }
        float bias=8.0*dot(x0+x1,vec4(1)),bias2=8.0*dot(y0+y1,vec4(1));
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            vec4 lo=unpackUnorm4x8(qs[r]&0x0f0f0f0fu)*255.0;
            vec4 hi=unpackUnorm4x8((qs[r]>>4)&0x0f0f0f0fu)*255.0;
            sums[r]=fma(dot(lo,x0)+dot(hi,x1)-bias,scale[r],sums[r]);
            lo=unpackUnorm4x8(qs2[r]&0x0f0f0f0fu)*255.0;
            hi=unpackUnorm4x8((qs2[r]>>4)&0x0f0f0f0fu)*255.0;
            sums[r]=fma(dot(lo,y0)+dot(hi,y1)-bias2,scale2[r],sums[r]);
        }
    }
#endif
    for(;b<blocks;b+=16) {
        vec4 x0=b_v4.v[b*8+part],x1=b_v4.v[b*8+part+4];
        float bias=8.0*dot(x0+x1,vec4(1));
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            uint block=(first+r)*blocks+b;
            float scale=uintBitsToFloat(ww.words[block]);
            uint qs=ww.words[p.stride+block*4+part];
            vec4 lo=unpackUnorm4x8(qs&0x0f0f0f0fu)*255.0;
            vec4 hi=unpackUnorm4x8((qs>>4)&0x0f0f0f0fu)*255.0;
            sums[r]=fma(dot(lo,x0)+dot(hi,x1)-bias,scale,sums[r]);
        }
    }
    [[unroll]] for(uint r=0;r<ROWS;r++) {
        float sum=subgroupAdd(sums[r]);
        if(lane==0&&first+r<p.n)c_f.v[first+r]=sum;
    }
}
