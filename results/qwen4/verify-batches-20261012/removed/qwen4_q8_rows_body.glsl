#include "common.glsl"
// Q8_0 matrix times C token columns (decode and verification batches), a lane per row: lane l of a wave walks row
// 64 * group + l over the wave's segment of K in whole 64-byte lines (two blocks), so every weight is decoded once
// for all C columns and no sums cross lanes; the columns' inputs are the same for every lane (scalar loads, SGPR
// operands). The KS waves of a workgroup take KS consecutive segments of the same rows and add their sums through LDS
// in wave order. Weights as qwen4_q8_mvw (quant::repack_q8h: f16 block scales, then 32-byte blocks from word
// p.stride, lines 64-byte aligned). b: x [t][k], c: y [t][n]. p.n rows, p.k columns (a multiple of 64 * KS).
// Workgroup y takes columns C * y ..
layout(local_size_x=KS*64) in;
layout(constant_id=0) const uint C=1;
layout(binding=0,std430) readonly restrict buffer WeightQuads { uvec4 q[]; } wq;
shared float red[KS][4][64];
vec4 i8(uint w){return vec4(ivec4(w<<24,w<<16,w<<8,w)>>24);}
void main(){
uint lane=gl_SubgroupInvocationID,wv=gl_SubgroupID,nb=batch(),t0=gl_WorkGroupID.y*C;
if(t0>=nb)return;
uint row=gl_WorkGroupID.x*64+lane,r=min(row,p.n-1),blocks=p.k/32,lines=p.k/(64*KS);
uint xo[4];[[unroll]]for(uint z=0;z<4;z++)xo[z]=min(t0+z,nb-1)*(p.k/4)+wv*lines*16;
float acc[4];[[unroll]]for(uint z=0;z<4;z++)acc[z]=0;
uint b0=r*blocks+wv*lines*2;
for(uint li=0;li<lines;li++){
    uint blk=b0+2*li,a=p.stride/4+blk*2;
    uvec4 q[4];q[0]=wq.q[a];q[1]=wq.q[a+1];q[2]=wq.q[a+2];q[3]=wq.q[a+3];
    vec2 d=unpackHalf2x16(ww.words[blk/2]);
    [[unroll]]for(uint h=0;h<2;h++){
        float s[4];[[unroll]]for(uint z=0;z<4;z++)s[z]=0;
        [[unroll]]for(uint j=0;j<8;j++){vec4 w=i8(q[h*2+j/4][j%4]);
            [[unroll]]for(uint z=0;z<C;z++){vec4 x=b_v4.v[xo[z]+li*16+h*8+j];s[z]=fma(w.x,x.x,s[z]);s[z]=fma(w.y,x.y,s[z]);
                s[z]=fma(w.z,x.z,s[z]);s[z]=fma(w.w,x.w,s[z]);}}
        [[unroll]]for(uint z=0;z<C;z++)acc[z]=fma(s[z],d[h],acc[z]);
    }
}
if(wv>0)[[unroll]]for(uint z=0;z<C;z++)red[wv][z][lane]=acc[z];
barrier();
if(wv==0&&row<p.n)[[unroll]]for(uint z=0;z<C;z++)if(t0+z<nb){float v=acc[z];
    for(uint k=1;k<KS;k++)v+=red[k][z][lane];
    c_f.v[(t0+z)*p.n+row]=v;}
}
