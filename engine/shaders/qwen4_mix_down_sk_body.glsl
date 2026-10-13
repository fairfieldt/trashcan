#include "common.glsl"
#include "qwen4_mix.glsl"
// First half of a hyper-connection mixer for small batches, split over the residual (qwen4_mix_down has 21
// workgroups for one token, too few to stream the matrix): workgroup (g, tg * C + c) takes 16 of the 324 rows (4 per
// wave) of COLS tokens from COLS * tg over chunk c of CB blocks (C = k / 32 / CB chunks, each inside one stream),
// reading the weights once for the tokens, with the pending combine x = x_in + w_prev * y applied on the fly and the
// stream gammas folded in, and writes the partial dot products and the chunk's sum of squares to the partials
// [t][c][m + 8] (sum of squares at m + 4). The last workgroup of the token group to finish (counted in extra, which
// it resets) sums the chunks, applies the stream norms, row = sum_s inv_s * W_s . (gamma_s * x_s), and writes
// qwen4_mix_down's output: bottleneck rows silu(row / 4), write rows 2 sigmoid(row / 4), the inverse norms.
// a: x [t][hc][n], b: y [t][n] (flags & 1: combine), c: partials, d: previous mixer out (write weights at m),
// e: gamma [hc][n], f: out [t][m + 8] (bottleneck | write | inv norms), extra: counters (u32 per token group).
// p.n = stream width, p.k = hc * n, p.m = bottleneck rows, p.stride = word offset of the blocks, p.r0 = eps bits.
#define CB 40
// Chunks per stream (a stream of 2560) and rows handled per invocation by the finishing workgroup (at most 512).
#define CPS 2
layout(set=2,binding=2,std430) coherent buffer Partials { float v[]; } cp;
layout(set=2,binding=7,std430) coherent buffer Counters { uint v[]; } cnt;
layout(local_size_x=256) in;
shared bool last;
void main(){
uint lane=gl_SubgroupInvocationID,blocks=p.k/32,sb=p.n/32,C=blocks/CB,nb=batch();
uint t0=(gl_WorkGroupID.y/C)*COLS,c=gl_WorkGroupID.y%C;
uint rows=p.m+4,first=gl_WorkGroupID.x*16+gl_SubgroupID*4,s=c*CB/sb,P=p.m+8;
bool comb=(p.flags&1)!=0;
float ws[COLS],ss[COLS],acc[4][COLS];
[[unroll]]for(uint z=0;z<COLS;z++){ws[z]=comb&&t0+z<nb?d_f.v[(t0+z)*P+p.m+s]:0.0;ss[z]=0;
    [[unroll]]for(uint r=0;r<4;r++)acc[r][z]=0;}
uint row[4];[[unroll]]for(uint r=0;r<4;r++)row[r]=min(first+r,rows-1);
if(FMT==0){uint part=lane%4;
    for(uint b=c*CB+lane/4;b<(c+1)*CB;b+=16){
        uint i0=(b*32-s*p.n)/4+part;
        uint q[4];float sc[4];
        [[unroll]]for(uint r=0;r<4;r++){uint blk=row[r]*blocks+b;q[r]=ww.words[p.stride+blk*4+part];sc[r]=wscale(blk);}
        vec4 e0=e_v4.v[s*p.n/4+i0],e1=e_v4.v[s*p.n/4+i0+4];
        [[unroll]]for(uint z=0;z<COLS;z++)if(t0+z<nb){uint t=t0+z,x0i=(t*p.k+b*32)/4+part;
            vec4 x0=a_v4.v[x0i],x1=a_v4.v[x0i+4];
            if(comb){x0=fma(vec4(ws[z]),b_v4.v[t*p.n/4+i0],x0);x1=fma(vec4(ws[z]),b_v4.v[t*p.n/4+i0+4],x1);}
            ss[z]+=dot(x0,x0)+dot(x1,x1);
            vec4 g0=x0*e0,g1=x1*e1;
            [[unroll]]for(uint r=0;r<4;r++)acc[r][z]=fma(dot(q4_lo(q[r]),g0)+dot(q4_hi(q[r]),g1),sc[r],acc[r][z]);}
    }
}else{uint part=lane%8;
    for(uint b=c*CB+lane/8;b<(c+1)*CB;b+=8){
        uint i0=(b*32-s*p.n)/4+part;
        uint q[4];float sc[4];
        [[unroll]]for(uint r=0;r<4;r++){uint blk=row[r]*blocks+b;q[r]=ww.words[p.stride+blk*8+part];sc[r]=wscale(blk);}
        vec4 e0=e_v4.v[s*p.n/4+i0];
        [[unroll]]for(uint z=0;z<COLS;z++)if(t0+z<nb){uint t=t0+z;
            vec4 x0=a_v4.v[(t*p.k+b*32)/4+part];
            if(comb)x0=fma(vec4(ws[z]),b_v4.v[t*p.n/4+i0],x0);
            ss[z]+=dot(x0,x0);
            vec4 g0=x0*e0;
            [[unroll]]for(uint r=0;r<4;r++){uint w=q[r];
                acc[r][z]=fma(dot(vec4(ivec4(w<<24,w<<16,w<<8,w)>>24),g0),sc[r],acc[r][z]);}}
    }
}
[[unroll]]for(uint z=0;z<COLS;z++)if(t0+z<nb){uint o=((t0+z)*C+c)*P;
    [[unroll]]for(uint r=0;r<4;r++){float v=subgroupAdd(acc[r][z]);if(lane==0&&first+r<rows)cp.v[o+first+r]=v;}
    if(gl_WorkGroupID.x==0&&gl_SubgroupID==0){float v=subgroupAdd(ss[z]);if(lane==0)cp.v[o+rows]=v;}}
memoryBarrierBuffer();
barrier();
uint tg=gl_WorkGroupID.y/C,tid=gl_LocalInvocationID.x;
if(tid==0)last=atomicAdd(cnt.v[tg],1u)==gl_NumWorkGroups.x*C-1u;
barrier();
if(!last)return;
memoryBarrierBuffer();
if(tid==0)cnt.v[tg]=0;
float eps=uintBitsToFloat(p.r0);
for(uint z=0;z<COLS;z++)if(t0+z<nb){uint t=t0+z,base=t*C*P,o=t*P;
    // All partials of a row are loaded before they are summed (CPS chunks per stream).
    float q[4][CPS],a[2][4][CPS];
    [[unroll]]for(uint s=0;s<4;s++)[[unroll]]for(uint i=0;i<CPS;i++){uint at=base+(s*CPS+i)*P;q[s][i]=cp.v[at+p.m+4];
        [[unroll]]for(uint h=0;h<2;h++)a[h][s][i]=cp.v[at+min(tid+256*h,rows-1)];}
    float inv[4];
    [[unroll]]for(uint s=0;s<4;s++){float qs=0;[[unroll]]for(uint i=0;i<CPS;i++)qs+=q[s][i];
        inv[s]=inversesqrt(qs/float(p.n)+eps);}
    [[unroll]]for(uint h=0;h<2;h++){uint j=tid+256*h;if(j<rows){float v=0;
        [[unroll]]for(uint s=0;s<4;s++){float as=0;[[unroll]]for(uint i=0;i<CPS;i++)as+=a[h][s][i];v+=inv[s]*as;}
        float y=v*0.25;f_f.v[o+j]=j<p.m?y/(1.0+exp(-y)):2.0/(1.0+exp(-y));}}
    if(tid<4)f_f.v[o+p.m+4+tid]=inv[tid];}
}
