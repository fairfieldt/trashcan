#include "common.glsl"
// EXPERIMENT (verify-batches): qwen4_q8_mvw_body with the column count exact (1-4); with PF the next iteration's
// weights, scales and inputs are loaded before the current iteration is computed; with ITERS (= p.k / 512) every
// iteration's weights and scales are loaded first (a wave's rows in whole bursts). ROWS rows per wave, WAVES waves
// per workgroup (defines). Specialization constant: COLS | PF << 3 | ITERS << 4.
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
layout(local_size_x=WAVES*64) in;
layout(constant_id=0) const uint SPEC=1;
const uint COLS=SPEC&7u;
const bool PF=(SPEC&8u)!=0u;
const uint ITERS=(SPEC>>4)&15u;
const uint IT=ITERS>0u?ITERS:1u;
vec4 i8(uint w){return vec4(ivec4(w<<24,w<<16,w<<8,w)>>24);}
void main(){uint nb=batch(),lane=gl_SubgroupInvocationID,part=lane%4,blocks=p.k/32;
uint first=gl_WorkGroupID.x*(ROWS*WAVES)+gl_SubgroupID*ROWS,t=gl_WorkGroupID.y*COLS;
if(t>=nb)return;
float s[ROWS][4];[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<4;z++)s[r][z]=0;
uint col[4];[[unroll]]for(uint z=0;z<4;z++)col[z]=min(t+z,nb-1)*(p.k/4)+part*2;
uint rb[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)rb[r]=min(first+r,p.n-1)*blocks;
if(ITERS>0u){
    uvec2 wa[IT][ROWS];uint da[IT][ROWS];
    [[unroll]]for(uint i=0;i<IT;i++)[[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+lane/4+16*i;
        wa[i][r]=ww2.v[p.stride/2+blk*4+part];da[i][r]=uint(wh.halves[blk]);}
    [[unroll]]for(uint i=0;i<IT;i++){uint b=lane/4+16*i;
        vec4 x[4][2];[[unroll]]for(uint z=0;z<COLS;z++){uint xo=col[z]+b*8;x[z][0]=b_v4.v[xo];x[z][1]=b_v4.v[xo+1];}
        [[unroll]]for(uint r=0;r<ROWS;r++){vec4 lo=i8(wa[i][r].x),hi=i8(wa[i][r].y);float dr=unpackHalf2x16(da[i][r]).x;
            [[unroll]]for(uint z=0;z<COLS;z++)s[r][z]=fma(dot(lo,x[z][0])+dot(hi,x[z][1]),dr,s[r][z]);}}
}else{
uvec2 wn[ROWS];uint dn[ROWS];vec4 xn[4][2];
uint b=lane/4;
if(PF){[[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;wn[r]=ww2.v[p.stride/2+blk*4+part];dn[r]=uint(wh.halves[blk]);}
    [[unroll]]for(uint z=0;z<COLS;z++){uint xo=col[z]+b*8;xn[z][0]=b_v4.v[xo];xn[z][1]=b_v4.v[xo+1];}}
for(;b<blocks;b+=16){
    uvec2 w[ROWS];uint d[ROWS];vec4 x[4][2];
    if(PF){w=wn;d=dn;x=xn;uint bn=min(b+16,blocks-1);
        [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+bn;wn[r]=ww2.v[p.stride/2+blk*4+part];dn[r]=uint(wh.halves[blk]);}
        [[unroll]]for(uint z=0;z<COLS;z++){uint xo=col[z]+bn*8;xn[z][0]=b_v4.v[xo];xn[z][1]=b_v4.v[xo+1];}
    }else{[[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;w[r]=ww2.v[p.stride/2+blk*4+part];d[r]=uint(wh.halves[blk]);}
        [[unroll]]for(uint z=0;z<COLS;z++){uint xo=col[z]+b*8;x[z][0]=b_v4.v[xo];x[z][1]=b_v4.v[xo+1];}}
    [[unroll]]for(uint r=0;r<ROWS;r++){vec4 lo=i8(w[r].x),hi=i8(w[r].y);float dr=unpackHalf2x16(d[r]).x;
        [[unroll]]for(uint z=0;z<COLS;z++)s[r][z]=fma(dot(lo,x[z][0])+dot(hi,x[z][1]),dr,s[r][z]);}
}
}
[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<COLS;z++){float v=subgroupAdd(s[r][z]);
    if(lane==0&&first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=v;}}
