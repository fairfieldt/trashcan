#include "common.glsl"
#include "qwen4_expert.glsl"
// Gate and up projections of the routed and shared experts with SwiGLU, for batches below 16 tokens. One workgroup
// per 2 * P gate/up row pairs of one (token, slot), in two groups of 160 lanes: lane hb of a group owns half-block hb
// of every row (16 weights, two of a 32-weight block's four nibble words) and keeps the 16 inputs they multiply in
// registers, so each row is one contiguous 1280-byte read by its group and a lane decodes a block scale once per 16
// weights. A group walks P row pairs, leaving each lane's dot products in shared memory, which the workgroup sums.
// Weights in qwen4_expert.glsl formats; expert e's rows start at e * 2 * ff: ff gate rows, then ff up rows, each p.k
// (= 2560) long. b: u [t][k], c: act [t][slots][ff], d: ids [t][slots] (u32). p.n = ff, p.m = slots, p.k = d,
// p.stride = word offset of the blocks.
#ifndef P
#define P 8
#endif
#define HB 160
#define TP (2 * HB / (4 * P))
layout(local_size_x=2*HB) in;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
// Rows padded by TP floats: the sums below read consecutive words in consecutive lanes.
shared float red[4 * P][HB + TP];
shared float red2[4 * P][TP];
// Scale of block blk (IQ4_XS: from its super-block's header; ib = blk & 7).
float hscale(uint blk,uvec2 hd,uint ib){
    if(FMT==2){uint ls=((hd.y>>(4u*ib))&15u)|(((hd.x>>(16u+2u*ib))&3u)<<4);
        return unpackHalf2x16(hd.x).x*float(int(ls)-32);}
    return unpackHalf2x16(uint(wh.halves[blk])).x;
}
void main(){
init_lut();
uint l=gl_LocalInvocationID.x,g=l/HB,hb=l%HB,b=hb/2,h=hb%2,ts=gl_WorkGroupID.y,t=ts/p.m;
if(t>=batch())return;
uint e=d_u.v[ts],blocks=p.k/32,ib=b&7u;
uint x4=t*(p.k/4)+b*8+2*h;
vec4 xa0=b_v4.v[x4],xa1=b_v4.v[x4+1],xb0=b_v4.v[x4+4],xb1=b_v4.v[x4+5];
// Gate row of the group's first row pair; the up row is p.n rows further.
uint i0=gl_WorkGroupID.x*2*P+g*P;
uint bg=(e*2*p.n+i0)*blocks+b,du=p.n*blocks;
uvec2 qg=ww2.v[p.stride/2+bg*2+h],qu=ww2.v[p.stride/2+(bg+du)*2+h];
uvec2 hg=uvec2(0),hu=uvec2(0);
if(FMT==2){hg=ww2.v[bg>>3];hu=ww2.v[(bg+du)>>3];}
[[dont_unroll]]for(uint j=0;j<P;j++){
    uvec2 cg=qg,cu=qu,ch=hg,cv=hu;uint cb=bg;
    if(j+1<P){bg+=blocks;qg=ww2.v[p.stride/2+bg*2+h];qu=ww2.v[p.stride/2+(bg+du)*2+h];
        if(FMT==2){hg=ww2.v[bg>>3];hu=ww2.v[(bg+du)>>3];}}
    float sg=hscale(cb,ch,ib),su=hscale(cb+du,cv,ib);
    float dg=dot(qlo(cg.x),xa0)+dot(qhi(cg.x),xb0)+dot(qlo(cg.y),xa1)+dot(qhi(cg.y),xb1);
    float dv=dot(qlo(cu.x),xa0)+dot(qhi(cu.x),xb0)+dot(qlo(cu.y),xa1)+dot(qhi(cu.y),xb1);
    red[(g*P+j)*2][hb]=dg*sg;red[(g*P+j)*2+1][hb]=dv*su;
}
barrier();
// Row slot rs = (group * P + j) * 2 + (0 gate, 1 up): TP lanes sum HB / TP partials each, then one lane per row
// pair sums those.
uint rs=l/TP,part=l%TP;
float s=0;
[[unroll]]for(uint i=0;i<HB/TP;i++)s+=red[rs][part+TP*i];
red2[rs][part]=s;
barrier();
if(l<2*P){float a=0,c=0;
    [[unroll]]for(uint i=0;i<TP;i++){a+=red2[2*l][i];c+=red2[2*l+1][i];}
    uint row=gl_WorkGroupID.x*2*P+l;
    if(row<p.n)c_f.v[ts*p.n+row]=a/(1.0+exp(-a))*c;}
}
