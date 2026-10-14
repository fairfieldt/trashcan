// QSA indexer block scores: score[t][b] = sum over heads of relu(q[t][h] . pooled[b]) / sqrt(dim) for the complete
// blocks b < nb(t) = (position + t + 1) / kind, only for tokens past the dense regime (nb(t) > p.m). A lane per
// block: it holds half of the block's pooled key at a time and computes every head's dot product with each of up to
// TT tokens' queries, which are uniform across the wave (scalar loads, or with QLDS from shared memory, where the
// latency is lower for a single token). The dot products keep the order of a wave
// reduction with lanes l holding dims l and l + 64 (q[l] k[l] + q[l+64] k[l+64], then a balanced tree over l: the
// swizzles of a GFX6 subgroupAdd), so the scores, and the blocks qwen4_idx_select picks, are the same bits as with a
// subgroupAdd per dot product.
// Grid: x = cap / kind / 256 block groups (blocks past the last token's nb exit), y = token groups of TT.
// a: indexer queries [t][heads][dim] (normed, rotated), b: pooled keys [dim / 4][cap / kind] (vec4s, qwen4_idx_pool),
// c: scores [t][cap / kind].
// p.heads = 4, p.dim = 128, p.kind, p.m, p.cap.
layout(local_size_x=256) in;
#define TT 8u
layout(set=2,binding=0,std430) readonly restrict buffer Queries { float v[]; } qs;
layout(set=2,binding=1,std430) readonly restrict buffer Pooled { vec4 v[]; } pk;
#ifdef QLDS
shared float qsh[TT*512u];
#define Q(x) qsh[(x)]
#else
#define Q(x) qs.v[t0*512u+(x)]
#endif
// KA: the registers holding the current half of the block's pooled key.
#define KA k
#define K0(i) KA[(i)>>2][(i)&3]
#define K1(i) KA[8+((i)>>2)][(i)&3]
// The replaced kernel's per-lane value as compiled for GFX6 (its fma lowered: two rounded products and their sum).
#define V(i) (Q(qb+(i))*K0(i)+Q(qb+64u+(i))*K1(i))
#define T2(i) (V(i)+V((i)+1))
#define T4(i) (T2(i)+T2((i)+2))
#define T8(i) (T4(i)+T4((i)+4))
#define T16(i) (T8(i)+T8((i)+8))
#define T32 (T16(0)+T16(16))
void main(){
uint b=gl_WorkGroupID.x*256u+gl_LocalInvocationID.x;
uint n=batch(),t0=gl_WorkGroupID.y*TT;if(t0>=n)return;
uint tn=min(TT,n-t0),stride=p.cap/p.kind,pos0=position()+t0;
uint nbmax=(pos0+tn)/p.kind;
// Nothing to score: every token of the group keeps all its blocks, or this group of blocks is not complete yet.
if(nbmax<=p.m||gl_WorkGroupID.x*256u>=nbmax)return;
uint bs=b%stride;
#ifdef QLDS
for(uint x=gl_LocalInvocationID.x;x<tn*512u;x+=256u)qsh[x]=qs.v[t0*512u+x];
barrier();
#endif
#ifdef QLDS
// Up to 8 tokens: both halves of the key at once and a loop over the heads, so that the code a wave runs once per
// token is small (a decode step fetches its instructions anew; unrolled over the heads, with the halves one after
// the other, this took 55-65 us per decode step at 8K-30K).
vec4 k[16],k2[16];
[[unroll]]for(uint j=0u;j<8u;j++){k[j]=pk.v[j*stride+bs];k[8u+j]=pk.v[(16u+j)*stride+bs];
    k2[j]=pk.v[(8u+j)*stride+bs];k2[8u+j]=pk.v[(24u+j)*stride+bs];}
[[unroll]]for(uint tt=0u;tt<TT;tt++)if(tt<tn){
    uint nb=(pos0+tt+1u)/p.kind;
    precise float s=0.0;
    [[dont_unroll]]for(uint h=0u;h<4u;h++){
        uint qb=(tt*4u+h)*128u;
        precise float lo=T32;
        qb+=32u;
#undef KA
#define KA k2
        precise float x=T32;
#undef KA
#define KA k
        s+=max(lo+x,0.0);
    }
    if(b<nb&&nb>p.m)c_f.v[(t0+tt)*stride+b]=s*inversesqrt(float(p.dim));
}
#else
precise float lo[TT*4u];
[[unroll]]for(uint part=0u;part<2u;part++){
    // Lanes part*32 .. part*32+31 of the reduction: dims part*32 + i and 64 + part*32 + i.
    vec4 k[16];
    [[unroll]]for(uint j=0u;j<8u;j++){k[j]=pk.v[(part*8u+j)*stride+bs];k[8u+j]=pk.v[(16u+part*8u+j)*stride+bs];}
    [[unroll]]for(uint tt=0u;tt<TT;tt++)if(tt<tn){
        uint nb=(pos0+tt+1u)/p.kind;
        // precise: no fused or reassociated arithmetic, so the sums keep the reduction's order.
        precise float s=0.0;
        [[unroll]]for(uint h=0u;h<4u;h++){
            uint qb=(tt*4u+h)*128u+part*32u;
            precise float x=T32;
            if(part==0u)lo[tt*4u+h]=x;else s+=max(lo[tt*4u+h]+x,0.0);
        }
        if(part==1u&&b<nb&&nb>p.m)c_f.v[(t0+tt)*stride+b]=s*inversesqrt(float(p.dim));
    }
}
#endif
}
