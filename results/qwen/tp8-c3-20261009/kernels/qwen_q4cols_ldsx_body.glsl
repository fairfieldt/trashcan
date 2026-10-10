#include "common.glsl"
// (round 3 experiment, not in the engine: microbench kernels qwen_q4c3x (ROWS 4), qwen_q4c3x_r3, qwen_q4c3x_r5 of
// trashcan-c3h; results/qwen/tp8-c3-20261009)
// COLS columns of a Q4_0 matrix product in the formulation of the one-column decode kernels, with the columns'
// activations staged in shared memory and read back per row: a workgroup loads the 16 blocks of each column that
// one iteration covers (COLS x 2 KB) cooperatively; each lane converts a row's nibbles once, then for every column
// reads its two vec4 of that column from the tile and accumulates, so the activations occupy 8 VGPRs instead of
// 8 per column. Each row's reads use a different zero offset from the push constants (p.r0, p.r1, ...), so the
// compiler cannot merge them across rows and hold every column again. The tile is loaded before the weights so
// that the wait for it does not cover the weights, whose latency then overlaps the barrier.
// Each column keeps the decode kernels' lane-to-block mapping, zero-point bias, FMA chain and subgroup reduction,
// so every column matches a one-column decode bit for bit. Weights in set 0 (p.stride: offset of the packed
// nibbles), x in b, y in c; p.n rows, p.k row length.
#define THREADS (WAVES*64)
#define TILE (COLS*128)
#define PER ((TILE+THREADS-1)/THREADS)
layout(binding=0,std430) readonly volatile buffer WeightWordsV { uint words[]; } wwv;
layout(local_size_x=THREADS) in;
// Per column: the first 64 vec4 hold the lanes' x0 (lane l at l), the next 64 their x1, so a wave's reads are
// contiguous.
shared vec4 tile[TILE];
void main() {
    uint lane=gl_SubgroupInvocationID,nb=batch(),t=gl_WorkGroupID.y*COLS,tid=gl_LocalInvocationID.x;
    uint first=gl_WorkGroupID.x*(ROWS*WAVES)+gl_SubgroupID*ROWS,part=lane%4,blocks=p.k/32;
    if(t>=nb)return;
    uint rofs[8]={0u,p.r0,p.r1,p.flags,p.offset,p.kind,p.window,p.cap};
    // Tile vec4 i (column i/128, block i%128/8, vec4 i%8 of the block) and its slot; every thread loads PER
    // vec4 (the index clamped into the tile and the vec4 into the column: a dead slot is never read).
    uint src[PER],lim[PER],dst[PER];
    [[unroll]] for(uint j=0;j<PER;j++) {
        uint i=min(tid+j*THREADS,TILE-1),z=i/128,g=i%128,q=g/8,v=g%8,col=min(t+z,nb-1)*(p.k/4);
        src[j]=col+q*8+v;
        lim[j]=col+p.k/4-1;
        dst[j]=z*128+(v<4?q*4+v:64+q*4+v-4);
    }
    float sums[COLS][ROWS];
    [[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++)sums[z][r]=0;
    for(uint b0=0;b0<blocks;b0+=16) {
        vec4 xt[PER];
        [[unroll]] for(uint j=0;j<PER;j++)xt[j]=b_v4.v[min(src[j]+b0*8,lim[j])];
        // This iteration's weights, clamped into the matrix for the lanes past the last block, whose results are
        // not used.
        uint b=b0+lane/4,bc=min(b,blocks-1),sb=first*blocks+bc,qb=p.stride+sb*4+part;
        float scale[ROWS];uint qs[ROWS];
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            scale[r]=uintBitsToFloat(wwv.words[sb+r*blocks]);
            qs[r]=wwv.words[qb+r*(blocks*4)];
        }
        [[unroll]] for(uint j=0;j<PER;j++)tile[dst[j]]=xt[j];
        barrier();
        if(b<blocks) {
            float bias[COLS];
            [[unroll]] for(uint r=0;r<ROWS;r++) {
                vec4 lo=unpackUnorm4x8(qs[r]&0x0f0f0f0fu)*255.0;
                vec4 hi=unpackUnorm4x8((qs[r]>>4)&0x0f0f0f0fu)*255.0;
                uint o=lane+rofs[r];
                [[unroll]] for(uint z=0;z<COLS;z++) {
                    vec4 x0=tile[z*128+o],x1=tile[z*128+64+o];
                    if(r==0)bias[z]=8.0*dot(x0+x1,vec4(1));
                    sums[z][r]=fma(dot(lo,x0)+dot(hi,x1)-bias[z],scale[r],sums[z][r]);
                }
            }
        }
        barrier();
    }
    [[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++) {
        float sum=subgroupAdd(sums[z][r]);
        if(lane==0&&first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=sum;
    }
}
