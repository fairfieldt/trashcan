#include "common.glsl"
#include "qwen4_expert.glsl"
#extension GL_KHR_shader_subgroup_quad : require
// Gate and up projections of the routed and shared experts with SwiGLU, for batches below 16 tokens. One workgroup
// per 16 rows of one (token, slot); each wave computes 4 gate rows and the matching 4 up rows over 5 steps of 16
// blocks: lane (bl, part) reads word part of block bl + 16 i of each row, as qwen4_moe_gu does, so each slot's result
// is the same bits. The 4 lanes of a block each decode the block scale of one row pair and share it (quad broadcast),
// and row offsets are constants of the 2560-column shape. PF 1: the next step's words, inputs and scale headers load
// while a step computes; PF 2: only its words. Weights in qwen4_expert.glsl formats; expert e's rows start at
// e * 2 * ff: ff gate rows, then ff up rows. b: u [t][k], c: act [t][slots][ff], d: ids [t][slots] (u32).
// p.n = ff, p.m = slots, p.k = d = 2560, p.stride = word offset of the blocks.
#ifndef PF
#define PF 1
#endif
#define BLOCKS 80
#define RW (BLOCKS * 4)
layout(local_size_x=256) in;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
float xscale(uvec2 hd,uint ib){
    uint ls=((hd.y>>(4u*ib))&15u)|(((hd.x>>(16u+2u*ib))&3u)<<4);
    return unpackHalf2x16(hd.x).x*float(int(ls)-32);
}
void main(){
init_lut();
uint lane=gl_SubgroupInvocationID,part=lane&3u,bl=lane>>2,w=gl_SubgroupID,ts=gl_WorkGroupID.y,t=ts/p.m;
if(t>=batch())return;
uint e=d_u.v[ts],first=gl_WorkGroupID.x*16+w*4,uoff=p.n*RW;
// Word of (gate row first, block bl, part); the lane's scale row is gate row first + part (and its up row).
uint wi=p.stride+((e*2*p.n+first)*BLOCKS+bl)*4+part;
uint sb=(e*2*p.n+first+part)*BLOCKS+bl,ub=p.n*BLOCKS,ib=bl&7u;
uint xi=t*640+bl*8+part;
vec4 g=vec4(0),u=vec4(0);
uint qg[4],qu[4];vec4 x0,x1;uvec2 hg,hu;
#if PF>=1
[[unroll]]for(uint r=0;r<4;r++){qg[r]=ww.words[wi+r*RW];qu[r]=ww.words[wi+uoff+r*RW];}
#endif
#if PF==1
x0=b_v4.v[xi];x1=b_v4.v[xi+4];
if(FMT==2){hg=ww2.v[sb>>3];hu=ww2.v[(sb+ub)>>3];}
#endif
[[dont_unroll]]for(uint i=0;i<BLOCKS/16;i++){
#if PF==1
    uint cg[4]=qg,cu[4]=qu;vec4 c0=x0,c1=x1;uvec2 dg=hg,du=hu;uint csb=sb;
    if(i+1<BLOCKS/16){wi+=64;sb+=16;xi+=128;
        [[unroll]]for(uint r=0;r<4;r++){qg[r]=ww.words[wi+r*RW];qu[r]=ww.words[wi+uoff+r*RW];}
        x0=b_v4.v[xi];x1=b_v4.v[xi+4];
        if(FMT==2){hg=ww2.v[sb>>3];hu=ww2.v[(sb+ub)>>3];}}
#elif PF==2
    uint cg[4]=qg,cu[4]=qu;vec4 c0=b_v4.v[xi],c1=b_v4.v[xi+4];uvec2 dg,du;uint csb=sb;
    if(FMT==2){dg=ww2.v[sb>>3];du=ww2.v[(sb+ub)>>3];}
    if(i+1<BLOCKS/16){wi+=64;[[unroll]]for(uint r=0;r<4;r++){qg[r]=ww.words[wi+r*RW];qu[r]=ww.words[wi+uoff+r*RW];}}
    sb+=16;xi+=128;
#else
    uint cg[4],cu[4];vec4 c0=b_v4.v[xi],c1=b_v4.v[xi+4];uvec2 dg,du;uint csb=sb;
    [[unroll]]for(uint r=0;r<4;r++){cg[r]=ww.words[wi+r*RW];cu[r]=ww.words[wi+uoff+r*RW];}
    if(FMT==2){dg=ww2.v[sb>>3];du=ww2.v[(sb+ub)>>3];}
    wi+=64;sb+=16;xi+=128;
#endif
    float sgm=FMT==2?xscale(dg,ib):unpackHalf2x16(uint(wh.halves[csb])).x;
    float sum=FMT==2?xscale(du,ib):unpackHalf2x16(uint(wh.halves[csb+ub])).x;
    [[unroll]]for(uint r=0;r<4;r++){
        float sg=subgroupQuadBroadcast(sgm,r),su=subgroupQuadBroadcast(sum,r);
        g[r]=fma(dot(qlo(cg[r]),c0)+dot(qhi(cg[r]),c1),sg,g[r]);
        u[r]=fma(dot(qlo(cu[r]),c0)+dot(qhi(cu[r]),c1),su,u[r]);
    }
}
// As qwen4_moe_gu sums them (the same per-lane sums, so the same bits).
g=subgroupAdd(g);u=subgroupAdd(u);
if(lane==0)[[unroll]]for(uint r=0;r<4;r++)if(first+r<p.n){float a=g[r];c_f.v[ts*p.n+first+r]=a/(1.0+exp(-a))*u[r];}
}
