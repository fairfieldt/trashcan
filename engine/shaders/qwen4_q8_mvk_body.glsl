#include "common.glsl"
// Q8_0 matrix times COLS = 1-4 token columns with each row's K split over two waves (the out projections: 2560 rows
// of 6144): a workgroup of 4 waves takes 2 ROWS rows, wave w the rows ROWS (w / 2).. over K half w % 2, so the waves are
// half as long and twice as many as qwen4_q8_mvw4's; the halves are added through LDS. Loads and arithmetic per lane as
// qwen4_q8_mvw4 (8-byte loads, four lanes per 32-byte block). b: x [t][k], c: y [t][n]. p.n rows, p.k columns (a
// multiple of 1024), p.stride = word offset of the blocks. Workgroup y takes columns COLS y ..
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
layout(local_size_x=256) in;
layout(constant_id=0) const uint COLS=1;
shared float half1[2][ROWS][4];
vec4 i8(uint w){return vec4(ivec4(w<<24,w<<16,w<<8,w)>>24);}
void main(){uint nb=batch(),lane=gl_SubgroupInvocationID,wv=gl_SubgroupID,part=lane%4,blocks=p.k/32,hb=blocks/2;
uint rg=wv/2,kh=wv%2,first=gl_WorkGroupID.x*(2*ROWS)+rg*ROWS,t=gl_WorkGroupID.y*COLS;
if(t>=nb)return;
float s[ROWS][4];[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<4;z++)s[r][z]=0;
uint col[4];[[unroll]]for(uint z=0;z<4;z++)col[z]=min(t+z,nb-1)*(p.k/4)+part*2;
uint rb[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)rb[r]=min(first+r,p.n-1)*blocks;
for(uint b=kh*hb+lane/4;b<(kh+1)*hb;b+=16){
    uvec2 w[ROWS];uint d[ROWS];
    [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;w[r]=ww2.v[p.stride/2+blk*4+part];d[r]=uint(wh.halves[blk]);}
    vec4 x[4][2];[[unroll]]for(uint z=0;z<COLS;z++){uint xo=col[z]+b*8;x[z][0]=b_v4.v[xo];x[z][1]=b_v4.v[xo+1];}
    [[unroll]]for(uint r=0;r<ROWS;r++){vec4 lo=i8(w[r].x),hi=i8(w[r].y);float dr=unpackHalf2x16(d[r]).x;
        [[unroll]]for(uint z=0;z<COLS;z++)s[r][z]=fma(dot(lo,x[z][0])+dot(hi,x[z][1]),dr,s[r][z]);}
}
float v[ROWS][4];
[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<COLS;z++){v[r][z]=subgroupAdd(s[r][z]);
    if(kh==1&&lane==0)half1[rg][r][z]=v[r][z];}
barrier();
if(kh==0&&lane==0)[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<COLS;z++)
    if(first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=v[r][z]+half1[rg][r][z];}
