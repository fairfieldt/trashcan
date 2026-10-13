#include "common.glsl"
// Q8_0 matrix times up to COLS token columns (decode and small batches) with 8-byte loads: four lanes per 32-byte
// block, each loading 8 weights as one uvec2, so a wave covers 16 blocks of a row per load; ROWS rows per wave,
// WAVES waves per workgroup. Weights: all f16 block scales, then the 32-byte blocks from word p.stride
// (quant::repack_q8h). b: x [t][k], c: y [t][n]. p.n rows, p.k columns. Several columns load their inputs first and
// decode each row's weights once for all of them (64 VGPRs for 3 columns against 84; the same sums).
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
layout(local_size_x=WAVES*64) in;
layout(constant_id=0) const uint COLS=1;
vec4 i8(uint w){return vec4(ivec4(w<<24,w<<16,w<<8,w)>>24);}
void main(){uint nb=batch(),lane=gl_SubgroupInvocationID,part=lane%4,blocks=p.k/32;
uint first=gl_WorkGroupID.x*(ROWS*WAVES)+gl_SubgroupID*ROWS,t=gl_WorkGroupID.y*COLS;
if(t>=nb)return;
float s[ROWS][4];[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<4;z++)s[r][z]=0;
if(COLS==1){
    uint col=t*(p.k/4);
    for(uint b=lane/4;b<blocks;b+=16){
        uvec2 w[ROWS];float d[ROWS];
        [[unroll]]for(uint r=0;r<ROWS;r++){uint row=min(first+r,p.n-1),blk=row*blocks+b;
            w[r]=ww2.v[p.stride/2+blk*4+part];d[r]=unpackHalf2x16(uint(wh.halves[blk])).x;}
        uint xo=col+b*8+part*2;vec4 x0=b_v4.v[xo],x1=b_v4.v[xo+1];
        [[unroll]]for(uint r=0;r<ROWS;r++)s[r][0]=fma(dot(i8(w[r].x),x0)+dot(i8(w[r].y),x1),d[r],s[r][0]);
    }
}else{
    uint col[4];[[unroll]]for(uint z=0;z<4;z++)col[z]=min(t+z,nb-1)*(p.k/4)+part*2;
    uint rb[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)rb[r]=min(first+r,p.n-1)*blocks;
    for(uint b=lane/4;b<blocks;b+=16){
        uvec2 w[ROWS];uint d[ROWS];
        [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;w[r]=ww2.v[p.stride/2+blk*4+part];d[r]=uint(wh.halves[blk]);}
        vec4 x[4][2];[[unroll]]for(uint z=0;z<COLS;z++){uint xo=col[z]+b*8;x[z][0]=b_v4.v[xo];x[z][1]=b_v4.v[xo+1];}
        [[unroll]]for(uint r=0;r<ROWS;r++){vec4 lo=i8(w[r].x),hi=i8(w[r].y);float dr=unpackHalf2x16(d[r]).x;
            [[unroll]]for(uint z=0;z<COLS;z++)s[r][z]=fma(dot(lo,x[z][0])+dot(hi,x[z][1]),dr,s[r][z]);}
    }
}
[[unroll]]for(uint r=0;r<ROWS;r++)[[unroll]]for(uint z=0;z<COLS;z++){float v=subgroupAdd(s[r][z]);
    if(lane==0&&first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=v;}}
