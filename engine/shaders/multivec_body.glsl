#include "common.glsl"
#ifdef MULTI_ROWS
#define ROWS 4
#if COLS == 2
#define SUM vec2
#elif COLS == 3
#define SUM vec3
#else
#define SUM vec4
#endif
layout(local_size_x=256) in;
// Issue all independent loads before decoding and accumulating. Keep the original
// dot products, per-lane FMA chains and subgroup reduction order.
void main(){uint nb=batch(),lane=gl_SubgroupInvocationID,first=gl_WorkGroupID.x*(4*ROWS)+gl_SubgroupID*ROWS,t=gl_WorkGroupID.y*COLS;
if(first>=p.n||t>=nb)return;SUM sums[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)sums[r]=SUM(0);
uint colbase[COLS];[[unroll]]for(uint z=0;z<COLS;z++)colbase[z]=min(t+z,nb-1)*(p.k/4);
if(p.kind==0){uint part=lane%4;
for(uint b=lane/4;b<p.k/32;b+=16){uint qs[ROWS];float d[ROWS];
[[unroll]]for(uint r=0;r<ROWS;r++){uint block=(first+r)*(p.k/32)+b;qs[r]=ww.words[p.stride+block*4+part];d[r]=uintBitsToFloat(ww.words[block]);}
vec4 a[COLS],c[COLS];[[unroll]]for(uint z=0;z<COLS;z++){uint x=colbase[z]+b*8+part;a[z]=b_v4.v[x];c[z]=b_v4.v[x+4];}
[[unroll]]for(uint r=0;r<ROWS;r++){vec4 lo=q4_lo(qs[r]),hi=q4_hi(qs[r]);
[[unroll]]for(uint z=0;z<COLS;z++)sums[r][z]=fma(dot(lo,a[z])+dot(hi,c[z]),d[r],sums[r][z]);}}
}else{
for(uint i=lane;i<p.k/4;i+=64){uint qs[ROWS];float d[ROWS];
[[unroll]]for(uint r=0;r<ROWS;r++){uint row=first+r;qs[r]=ww.words[p.stride+row*(p.k/4)+i];d[r]=uintBitsToFloat(ww.words[row*(p.k/16)+i/4]);}
vec4 x[COLS];[[unroll]]for(uint z=0;z<COLS;z++)x[z]=b_v4.v[colbase[z]+i];
[[unroll]]for(uint r=0;r<ROWS;r++){uint v=qs[r];vec4 q=vec4(ivec4(v<<24,v<<16,v<<8,v)>>24);
[[unroll]]for(uint z=0;z<COLS;z++)sums[r][z]=fma(dot(q,x[z]),d[r],sums[r][z]);}}
}
[[unroll]]for(uint r=0;r<ROWS;r++){SUM v=subgroupAdd(sums[r]);if(lane==0&&first+r<p.n)[[unroll]]for(uint z=0;z<COLS;z++)if(t+z<nb)c_f.v[(t+z)*p.n+first+r]=v[z];}}

#else
layout(local_size_x=256) in;
void main(){uint nb=batch(),lane=gl_SubgroupInvocationID,row=gl_WorkGroupID.x*4+gl_SubgroupID;
uint t=gl_WorkGroupID.y*COLS;if(row>=p.n||t>=nb)return;
float sums[COLS];[[unroll]]for(uint z=0;z<COLS;z++)sums[z]=0;
if(p.kind==0){
uint part=lane%4;
for(uint b=lane/4;b<p.k/32;b+=16){uint block=row*(p.k/32)+b;
float scale=uintBitsToFloat(ww.words[block]);uint qs=ww.words[p.stride+block*4+part];
vec4 lo=q4_lo(qs);
vec4 hi=q4_hi(qs);
[[unroll]]for(uint z=0;z<COLS;z++)if(t+z<nb){uint x=(t+z)*(p.k/4)+b*8+part;
sums[z]=fma(dot(lo,b_v4.v[x])+dot(hi,b_v4.v[x+4]),scale,sums[z]);}
}
}else{
for(uint i=lane;i<p.k/4;i+=64){uint idx=row*(p.k/4)+i;uint v=ww.words[p.stride+idx];
ivec4 qs=ivec4(v<<24,v<<16,v<<8,v)>>24;float scale=uintBitsToFloat(ww.words[row*(p.k/16)+i/4]);
[[unroll]]for(uint z=0;z<COLS;z++)if(t+z<nb)sums[z]=fma(dot(vec4(qs),b_v4.v[(t+z)*(p.k/4)+i]),scale,sums[z]);}
}
[[unroll]]for(uint z=0;z<COLS;z++){float v=subgroupAdd(sums[z]);if(lane==0&&t+z<nb)c_f.v[(t+z)*p.n+row]=v;}
}

#endif
