#include "common.glsl"
#include "qwen4_expert.glsl"
// Down projections of a token's experts, summed with the routing weights, for batches below 16 tokens:
// y[t][i] = sum_j w[t][j] * W_e(j)[i] . act[t][j]. One workgroup per R output rows of one token: lane l < slots * 40
// owns half-block l % 40 of slot l / 40 (16 weights of a 640-wide row) and keeps the 16 inputs they multiply in
// registers; per row it leaves its term (dot * scale * weight, as qwen4_moe_down_w forms it) in shared memory, and
// the workgroup sums each row's terms. Formats 0 and 1 of qwen4_expert.glsl (an f16 scale per block); expert e's rows
// start at e * p.n, each p.k (= 640) long. b: act [t][slots][ff], c: y [t][n], d: ids [t][slots] (u32),
// e: weights [t][slots]. p.n = d, p.m = slots (at most 11), p.k = ff, p.stride = word offset of the blocks.
#ifndef R
#define R 8
#endif
#define HB 40
#define NL 448
#define TP (NL / R)
layout(local_size_x=NL) in;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
// Rows padded by TP floats: the sums below read consecutive words in consecutive lanes.
shared float red[R][NL + TP];
shared float red2[R][TP];
void main(){
init_lut();
uint l=gl_LocalInvocationID.x,t=gl_WorkGroupID.y;
if(t>=batch())return;
bool on=l<p.m*HB;
uint s=min(l/HB,p.m-1),hb=l%HB,b=hb/2,h=hb%2,blocks=p.k/32;
uint e=d_u.v[t*p.m+s];
float wt=e_f.v[t*p.m+s];
uint x4=(t*p.m+s)*(p.k/4)+b*8+2*h;
vec4 xa0=b_v4.v[x4],xa1=b_v4.v[x4+1],xb0=b_v4.v[x4+4],xb1=b_v4.v[x4+5];
uint r0=gl_WorkGroupID.x*R;
uint blk=(e*p.n+min(r0,p.n-1))*blocks+b;
uvec2 q=ww2.v[p.stride/2+blk*2+h];
float sc=unpackHalf2x16(uint(wh.halves[blk])).x;
[[dont_unroll]]for(uint j=0;j<R;j++){
    uvec2 cq=q;float cs=sc;
    if(j+1<R){blk=(e*p.n+min(r0+j+1,p.n-1))*blocks+b;q=ww2.v[p.stride/2+blk*2+h];sc=unpackHalf2x16(uint(wh.halves[blk])).x;}
    float d=dot(qlo(cq.x),xa0)+dot(qlo(cq.y),xa1)+dot(qhi(cq.x),xb0)+dot(qhi(cq.y),xb1);
    red[j][l]=on?d*(cs*wt):0.0;
}
barrier();
// Row j's NL terms: TP lanes sum R each, then one lane per row sums those.
uint j=l/TP,part=l%TP;
float a=0;
[[unroll]]for(uint i=0;i<R;i++)a+=red[j][part+TP*i];
red2[j][part]=a;
barrier();
if(l<R&&r0+l<p.n){float v=0;
    [[unroll]]for(uint i=0;i<TP;i++)v+=red2[l][i];
    c_f.v[t*p.n+r0+l]=v;}
}
