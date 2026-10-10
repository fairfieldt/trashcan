#include "common.glsl"
// (round 3 experiment, not in the engine: microbench kernels qwen_q4c3l (DOUBLE 0), qwen_q4c3l2 (DOUBLE 1),
// qwen_q4c3l_k5120, qwen_q4c3l2_k5120 of trashcan-c3c; results/qwen/tp8-c3-20261009)
// COLS columns of a Q4_0 matrix product in the formulation of the one-column decode kernels (matvecq4,
// matvecq4_r2), with the columns' activations staged in shared memory: a workgroup loads the 16 blocks of each
// column that one iteration covers (COLS x 2 KB) cooperatively, and every lane reads its two vec4 of a column from
// the tile right before it uses them, so the activations occupy 8 VGPRs instead of 8 per column. The packed
// nibbles are converted again for every column (the mask of column z is XORed with a push constant that is zero,
// p.r0/p.r1/p.flags, so the compiler cannot merge the conversions and hold every row's bytes across the columns),
// which keeps the kernel under the 64-VGPR occupancy step of the register-resident kernel (qwen_q4cols_body).
// The weights are read through a volatile view so that their loads stay at the top of the iteration, ahead of
// the tile load and the barrier, instead of being sunk to their first use. With DOUBLE the tile is double-buffered:
// the next iteration's activations are loaded while this one computes, and one barrier per iteration remains.
// Each column keeps the decode kernels' lane-to-block mapping, zero-point bias, FMA chain and subgroup reduction,
// so every column matches a one-column decode bit for bit (the row layout per wave does not change a row's
// arithmetic). Weights in set 0 (p.stride: offset of the packed nibbles), x in b, y in c; p.n rows, p.k row length.
#ifdef KFIX
#define KLEN KFIX
#else
#define KLEN p.k
#endif
#ifndef DOUBLE
#define DOUBLE 0
#endif
#define THREADS (WAVES*64)
#define TILE (COLS*128)
#define PER ((TILE+THREADS-1)/THREADS)
layout(binding=0,std430) readonly volatile buffer WeightWordsV { uint words[]; } wwv;
layout(local_size_x=THREADS) in;
// Per column: the first 64 vec4 hold the lanes' x0 (lane l at l), the next 64 their x1, so a wave's reads are
// contiguous.
shared vec4 tile[(DOUBLE+1)*TILE];
void main() {
    uint lane=gl_SubgroupInvocationID,nb=batch(),t=gl_WorkGroupID.y*COLS,tid=gl_LocalInvocationID.x;
    uint first=gl_WorkGroupID.x*(ROWS*WAVES)+gl_SubgroupID*ROWS,part=lane%4,blocks=KLEN/32;
    if(t>=nb)return;
    uint mask[4]={0x0f0f0f0fu,0x0f0f0f0fu^p.r0,0x0f0f0f0fu^p.r1,0x0f0f0f0fu^p.flags};
    // Tile vec4 i (column i/128, block i%128/8, vec4 i%8 of the block) and its slot; every thread loads PER
    // vec4 (the index clamped into the tile and the vec4 into the column: a dead slot is never read).
    uint src[PER],lim[PER],dst[PER];
    [[unroll]] for(uint j=0;j<PER;j++) {
        uint i=min(tid+j*THREADS,TILE-1),z=i/128,g=i%128,q=g/8,v=g%8,col=min(t+z,nb-1)*(KLEN/4);
        src[j]=col+q*8+v;
        lim[j]=col+KLEN/4-1;
        dst[j]=z*128+(v<4?q*4+v:64+q*4+v-4);
    }
    float sums[COLS][ROWS];
    [[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++)sums[z][r]=0;
#if DOUBLE
    [[unroll]] for(uint j=0;j<PER;j++)tile[dst[j]]=b_v4.v[src[j]];
    barrier();
#endif
    for(uint b0=0;b0<blocks;b0+=16) {
        // This iteration's weights first, clamped into the matrix for the lanes past the last block, whose
        // results are not used.
        uint b=b0+lane/4,bc=min(b,blocks-1);
        float scale[ROWS];uint qs[ROWS];
        [[unroll]] for(uint r=0;r<ROWS;r++) {
            uint block=(first+r)*blocks+bc;
            scale[r]=uintBitsToFloat(wwv.words[block]);
            qs[r]=wwv.words[p.stride+block*4+part];
        }
#if DOUBLE
        uint buf=(b0/16)%2*TILE;
        vec4 next[PER];
        [[unroll]] for(uint j=0;j<PER;j++)next[j]=b_v4.v[min(src[j]+(b0+16)*8,lim[j])];
#else
        uint buf=0;
        [[unroll]] for(uint j=0;j<PER;j++)tile[dst[j]]=b_v4.v[min(src[j]+b0*8,lim[j])];
        barrier();
#endif
        if(b<blocks) {
            [[unroll]] for(uint z=0;z<COLS;z++) {
                vec4 x0=tile[buf+z*128+lane],x1=tile[buf+z*128+64+lane];
                float bias=8.0*dot(x0+x1,vec4(1));
                [[unroll]] for(uint r=0;r<ROWS;r++) {
                    vec4 lo=unpackUnorm4x8(qs[r]&mask[z])*255.0;
                    vec4 hi=unpackUnorm4x8((qs[r]>>4)&mask[z])*255.0;
                    sums[z][r]=fma(dot(lo,x0)+dot(hi,x1)-bias,scale[r],sums[z][r]);
                }
            }
        }
#if DOUBLE
        [[unroll]] for(uint j=0;j<PER;j++)tile[(TILE-buf)+dst[j]]=next[j];
#endif
        barrier();
    }
    [[unroll]] for(uint z=0;z<COLS;z++)[[unroll]] for(uint r=0;r<ROWS;r++) {
        float sum=subgroupAdd(sums[z][r]);
        if(lane==0&&first+r<p.n&&t+z<nb)c_f.v[(t+z)*p.n+first+r]=sum;
    }
}
