// EXPERIMENT copy of qwen4_mix_down_skw: ROWS rows per wave (define), NOFIN (spec bit 3) skips the finishing pass.
#include "common.glsl"
// First half of a Q8_0 hyper-connection mixer for C = 1-4 tokens per workgroup (batches of 1-8 tokens), split over
// the four streams of the residual: workgroup (g, tg * hc + s) takes 16 of the 324 rows (4 per wave) of tokens
// C tg.. over stream s, four lanes per 32-weight block loading 8 weights each (16 blocks of a row per load), reading
// the weights once for the C tokens, with the pending combine x = x_in + w_prev * y applied on the fly and the stream
// gammas folded in, and writes the partial dot products and the stream's sum of squares to the partials
// [t][s][m + 8] (sum of squares at m + 4). The last workgroup of the token group to finish (counted in extra, which
// it resets) applies the stream norms, row = sum_s inv_s * W_s . (gamma_s * x_s), and writes qwen4_mix_down's output:
// bottleneck rows silu(row / 4), write rows 2 sigmoid(row / 4), the inverse norms. (qwen4_mix_down_sk splits the
// residual in 8 chunks with 8 lanes per block; this layout reads the matrix once for 2, 3 and 4 tokens in 67, 83 and
// 105 us against 97, 124 and 149 for qwen4_mix_down_sk4, and as fast for one token, 50 us; one layout for every batch
// size keeps a token's results the same bits in any batch: results/qwen4/verify-batches-20261012.)
// a: x [t][hc][n], b: y [t][n] (flags & 1: combine), c: partials, d: previous mixer out (write weights at m),
// e: gamma [hc][n], f: out [t][m + 8] (bottleneck | write | inv norms), extra: counters (u32 per token group).
// p.n = stream width, p.k = hc * n, p.m = bottleneck rows, p.stride = word offset of the blocks, p.r0 = eps bits.
layout(constant_id=0) const uint SPEC=1;
const uint C=SPEC&7u;
const bool NOFIN=(SPEC&8u)!=0u;
layout(set=2,binding=2,std430) coherent buffer Partials { float v[]; } cp;
layout(set=2,binding=7,std430) coherent buffer Counters { uint v[]; } cnt;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
layout(local_size_x=256) in;
shared bool last;
vec4 i8(uint w){return vec4(ivec4(w<<24,w<<16,w<<8,w)>>24);}
void main(){
uint lane=gl_SubgroupInvocationID,blocks=p.k/32,sb=p.n/32,nb=batch(),part=lane%4;
uint tg=gl_WorkGroupID.y/4,s=gl_WorkGroupID.y%4,t0=tg*C;
uint rows=p.m+4,first=gl_WorkGroupID.x*(4*ROWS)+gl_SubgroupID*ROWS,P=p.m+8;
bool comb=(p.flags&1)!=0;
float ws[4],ss[4],acc[ROWS][4];
uint tt[4];
[[unroll]]for(uint z=0;z<4;z++){tt[z]=min(t0+z,nb-1);ws[z]=comb?d_f.v[tt[z]*P+p.m+s]:0.0;ss[z]=0;
    [[unroll]]for(uint r=0;r<ROWS;r++)acc[r][z]=0;}
uint rb[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)rb[r]=min(first+r,rows-1)*blocks;
for(uint b=s*sb+lane/4;b<(s+1)*sb;b+=16){
    uint i0=(b*32-s*p.n)/4+part*2;
    uvec2 q[ROWS];uint d[ROWS];
    [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;q[r]=ww2.v[p.stride/2+blk*4+part];d[r]=uint(wh.halves[blk]);}
    vec4 e0=e_v4.v[s*p.n/4+i0],e1=e_v4.v[s*p.n/4+i0+1];
    [[unroll]]for(uint z=0;z<C;z++){uint xo=(tt[z]*p.k+b*32)/4+part*2;
        vec4 x0=a_v4.v[xo],x1=a_v4.v[xo+1];
        if(comb){x0=fma(vec4(ws[z]),b_v4.v[tt[z]*p.n/4+i0],x0);x1=fma(vec4(ws[z]),b_v4.v[tt[z]*p.n/4+i0+1],x1);}
        ss[z]+=dot(x0,x0)+dot(x1,x1);
        vec4 g0=x0*e0,g1=x1*e1;
        [[unroll]]for(uint r=0;r<ROWS;r++)acc[r][z]=fma(dot(i8(q[r].x),g0)+dot(i8(q[r].y),g1),unpackHalf2x16(d[r]).x,acc[r][z]);}
}
[[unroll]]for(uint z=0;z<C;z++)if(t0+z<nb){uint o=((t0+z)*4+s)*P;
    [[unroll]]for(uint r=0;r<ROWS;r++){float v=subgroupAdd(acc[r][z]);if(lane==0&&first+r<rows)cp.v[o+first+r]=v;}
    if(gl_WorkGroupID.x==0&&gl_SubgroupID==0){float v=subgroupAdd(ss[z]);if(lane==0)cp.v[o+rows]=v;}}
if(NOFIN)return;
memoryBarrierBuffer();
barrier();
uint tid=gl_LocalInvocationID.x;
if(tid==0)last=atomicAdd(cnt.v[tg],1u)==gl_NumWorkGroups.x*4u-1u;
barrier();
if(!last)return;
memoryBarrierBuffer();
if(tid==0)cnt.v[tg]=0;
float eps=uintBitsToFloat(p.r0);
for(uint z=0;z<C;z++)if(t0+z<nb){uint t=t0+z,base=t*4*P,o=t*P;
    // All partials of a row are loaded before they are summed.
    float q[4],a[2][4];
    [[unroll]]for(uint s=0;s<4;s++){uint at=base+s*P;q[s]=cp.v[at+p.m+4];
        [[unroll]]for(uint h=0;h<2;h++)a[h][s]=cp.v[at+min(tid+256*h,rows-1)];}
    float inv[4];
    [[unroll]]for(uint s=0;s<4;s++)inv[s]=inversesqrt(q[s]/float(p.n)+eps);
    [[unroll]]for(uint h=0;h<2;h++){uint j=tid+256*h;if(j<rows){float v=0;
        [[unroll]]for(uint s=0;s<4;s++)v+=inv[s]*a[h][s];
        float y=v*0.25;f_f.v[o+j]=j<p.m?y/(1.0+exp(-y)):2.0/(1.0+exp(-y));}}
    if(tid<4)f_f.v[o+p.m+4+tid]=inv[tid];}
}
