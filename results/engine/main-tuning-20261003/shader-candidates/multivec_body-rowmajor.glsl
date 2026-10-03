#include "common.glsl"
#ifdef MULTI_ROWS
#define ROWS 4
layout(local_size_x=256) in;
void main(){uint nb=batch(),lane=gl_SubgroupInvocationID,first=gl_WorkGroupID.x*(4*ROWS)+gl_SubgroupID*ROWS,t=gl_WorkGroupID.y*4;
if(first>=p.n||t>=nb)return;vec4 sums[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)sums[r]=vec4(0);
if(p.kind==0){uint part=lane%4;
for(uint b=lane/4;b<p.k/32;b+=16){
// Decode one row at a time to reduce live quantized values from 32 to eight.
// Activations are reloaded across rows and should be served by the vector cache.
[[unroll]]for(uint r=0;r<ROWS;r++){
uint block=(first+r)*(p.k/32)+b,v=ww.words[p.stride+block*4+part];
float d=uintBitsToFloat(ww.words[block]);vec4 lo=q4_lo(v),hi=q4_hi(v);
[[unroll]]for(uint z=0;z<4;z++)if(t+z<nb){uint x=(t+z)*(p.k/4)+b*8+part;
sums[r][z]=fma(dot(lo,b_v4.v[x])+dot(hi,b_v4.v[x+4]),d,sums[r][z]);}
}}
}else{
for(uint i=lane;i<p.k/4;i+=64){vec4 q[ROWS];float d[ROWS];
[[unroll]]for(uint r=0;r<ROWS;r++){uint row=first+r,v=ww.words[p.stride+row*(p.k/4)+i];q[r]=vec4(ivec4(v<<24,v<<16,v<<8,v)>>24);d[r]=uintBitsToFloat(ww.words[row*(p.k/16)+i/4]);}
[[unroll]]for(uint z=0;z<4;z++)if(t+z<nb){vec4 x=b_v4.v[(t+z)*(p.k/4)+i];[[unroll]]for(uint r=0;r<ROWS;r++)sums[r][z]=fma(dot(q[r],x),d[r],sums[r][z]);}}
}
[[unroll]]for(uint r=0;r<ROWS;r++){vec4 v=subgroupAdd(sums[r]);if(lane==0&&first+r<p.n)[[unroll]]for(uint z=0;z<4;z++)if(t+z<nb)c_f.v[(t+z)*p.n+first+r]=v[z];}}

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
