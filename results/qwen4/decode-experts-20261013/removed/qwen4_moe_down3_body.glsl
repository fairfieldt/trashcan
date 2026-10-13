#include "common.glsl"
#include "qwen4_expert.glsl"
// Down projections of a token's experts, summed with the routing weights, for batches below 16 tokens:
// y[t][i] = sum_j w[t][j] * W_e(j)[i] . act[t][j]. One workgroup per 16 output rows and token; each wave owns 4 rows
// and walks the token's (slot, block) pairs over its 32 lane pairs (two lanes per 32-weight block, each loading half
// of its nibbles as one uvec2), as qwen4_moe_down_w does, so the sums are the same bits; the pair walk is incremental
// and row offsets are constants of 640-wide rows. PF 2: the next pair's words and scales load while a pair computes.
// Formats 0 and 1 of qwen4_expert.glsl. b: act [t][slots][ff], c: y [t][n], d: ids [t][slots] (u32),
// e: weights [t][slots]. p.n = d, p.m = slots, p.k = ff = 640, p.stride = word offset of the blocks.
#ifndef PF
#define PF 0
#endif
#define BLOCKS 20
layout(local_size_x=256) in;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
void main(){
init_lut();
uint lane=gl_SubgroupInvocationID,h=lane&1u,w=gl_SubgroupID,t=gl_WorkGroupID.y;
if(t>=batch())return;
uint first=gl_WorkGroupID.x*16+w*4,pairs=p.m*BLOCKS;
// Pair f = lane / 2 + 32 i: slot f / 20, block f % 20.
uint f=lane>>1,j=f/BLOCKS,b=f%BLOCKS;
vec4 acc=vec4(0);
#if PF==2
uint blk=(d_u.v[t*p.m+j]*p.n+first)*BLOCKS+b;
uvec2 q[4];uint sc[4];
[[unroll]]for(uint r=0;r<4;r++){q[r]=ww2.v[p.stride/2+(blk+r*BLOCKS)*2+h];sc[r]=uint(wh.halves[blk+r*BLOCKS]);}
#endif
for(uint i=0;i*32<pairs;i++){
    bool on=f<pairs;uint jj=min(j,p.m-1),xo=(t*p.m+jj)*(p.k/4)+b*8+h*2;
    float cw=on?e_f.v[t*p.m+jj]:0.0;
    vec4 c0=b_v4.v[xo],c1=b_v4.v[xo+1],c2=b_v4.v[xo+4],c3=b_v4.v[xo+5];
#if PF==2
    uvec2 cq[4]=q;uint cs[4]=sc;
    f+=32;b+=12;j+=1;if(b>=BLOCKS){b-=BLOCKS;j++;}
    if(i*32+32<pairs){blk=(d_u.v[t*p.m+min(j,p.m-1)]*p.n+first)*BLOCKS+b;
        [[unroll]]for(uint r=0;r<4;r++){q[r]=ww2.v[p.stride/2+(blk+r*BLOCKS)*2+h];sc[r]=uint(wh.halves[blk+r*BLOCKS]);}}
    [[unroll]]for(uint r=0;r<4;r++)
        acc[r]=fma(dot(qlo(cq[r].x),c0)+dot(qlo(cq[r].y),c1)+dot(qhi(cq[r].x),c2)+dot(qhi(cq[r].y),c3),
            unpackHalf2x16(cs[r]).x*cw,acc[r]);
#else
    uint blk=(d_u.v[t*p.m+jj]*p.n+first)*BLOCKS+b;
    [[unroll]]for(uint r=0;r<4;r++){uvec2 q=ww2.v[p.stride/2+(blk+r*BLOCKS)*2+h];
        acc[r]=fma(dot(qlo(q.x),c0)+dot(qlo(q.y),c1)+dot(qhi(q.x),c2)+dot(qhi(q.y),c3),
            unpackHalf2x16(uint(wh.halves[blk+r*BLOCKS])).x*cw,acc[r]);}
    f+=32;b+=12;j+=1;if(b>=BLOCKS){b-=BLOCKS;j++;}
#endif
}
// As qwen4_moe_down_w sums them (the same per-lane sums, so the same bits).
acc=subgroupAdd(acc);
if(lane==0)[[unroll]]for(uint r=0;r<4;r++)if(first+r<p.n)c_f.v[t*p.n+first+r]=acc[r];
}
