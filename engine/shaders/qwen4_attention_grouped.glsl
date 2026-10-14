// QSA attention for prefill chunks, grouped by KV head: one workgroup per (KV head, token) serves the G = 12 query
// heads that share the head's K/V, so each K/V row of the token's visible positions (the selected complete blocks of
// p.kind positions in block order, then the incomplete tail ending at the token) is read once instead of once per
// query head. The four waves take turns of 64 positions. In a turn a lane takes one position and scores its K row
// against the 12 queries (uniform across the wave, so they come through scalar loads), the wave's running maximum per
// head moves on (online softmax, one cross-lane maximum per head per turn; the sums stay per lane until the end), and
// then the lanes take 4 dims each and add the 64 positions' V rows times their probabilities into 12 x 4
// accumulators. The four waves' partial results merge in shared memory at the end. (Six heads per workgroup, three
// waves per SIMD instead of two, took twice as long: every K/V row read and converted twice.)
// a: q (pre-scaled) [t][heads][dim], b: K cache, c: V cache, d: output [t][heads][dim], f: selected blocks [t][p.m]
// (u32). K/V caches: KVQ 0: f16 rows [cap][kvheads][dim]; 1: Q8 K and V (qwen4_kv_store: dim/4 words of int8, then
// dim/32 f16 scales); 2: f16 K, Q8 V. p.heads, p.kvheads, p.dim = 256, p.cap, p.kind, p.m; p.heads / p.kvheads = 12.
layout(local_size_x=256) in;
#ifndef KVQ
#define KVQ 0
#endif
#define G 12u
#define DIM 256u
// Head quads of a position's probabilities.
#define GQ ((G+3u)/4u)
// Q8 rows: DIM/4 words of int8 values, then DIM/64 words of f16 scales (one per 32 values).
#define QROW (DIM/4u+DIM/64u)
layout(set=2,binding=0,std430) readonly restrict buffer Queries { float v[]; } qs;
layout(set=2,binding=1,std430) readonly restrict buffer KRows { uvec4 v[]; } krow;
layout(set=2,binding=2,std430) readonly restrict buffer VRows { uvec2 v[]; } vrow;
layout(set=2,binding=2,std430) readonly restrict buffer VWords { uint v[]; } vword;
layout(set=2,binding=5,std430) readonly restrict buffer Selected { uint v[]; } selb;
// Per wave: the probabilities of its turn's 64 positions ([position][head quad]); after the loop, the merged output
// [head][64 lanes] (4 dims each).
shared vec4 buf[4u*64u*GQ];
shared uint skey[4u*64u];
shared float sm[4u*G],sl[4u*G];

void main(){
uint kh=gl_WorkGroupID.x,t=gl_WorkGroupID.y;
uint lane=gl_SubgroupInvocationID,w=gl_SubgroupID;
if(t>=batch())return;
uint pos=position()+t,nb=(pos+1u)/p.kind,nsel=min(nb,p.m),selpos=nsel*p.kind,total=selpos+pos+1u-nb*p.kind;
uint q0=(t*p.heads+kh*G)*DIM;
float M[G],L[G];vec4 o[G];
[[unroll]]for(uint h=0u;h<G;h++){M[h]=-1e30;L[h]=0.0;o[h]=vec4(0.0);}
uint turns=(total+63u)/64u;
for(uint c=w;c<turns;c+=4u){
    uint i=c*64u+lane;bool valid=i<total;
    uint key=i<selpos?selb.v[t*p.m+i/p.kind]*p.kind+i%p.kind:nb*p.kind+(i-selpos);
    key=valid?key%p.cap:0u;
    float s[G];[[unroll]]for(uint h=0u;h<G;h++)s[h]=0.0;
#if KVQ==1
    uint kr=(key*p.kvheads+kh)*(QROW/4u);
    uvec4 sc4=krow.v[kr+DIM/16u];
    for(uint g=0u;g<DIM/32u;g++){
        float part[G];[[unroll]]for(uint h=0u;h<G;h++)part[h]=0.0;
        [[unroll]]for(uint u=0u;u<2u;u++){
            uvec4 kk=krow.v[kr+g*2u+u];
            [[unroll]]for(uint j=0u;j<4u;j++){
                int word=int(kk[j]);
                [[unroll]]for(uint b=0u;b<4u;b++){
                    float kv=float(bitfieldExtract(word,int(b*8u),8));uint d=g*32u+u*16u+j*4u+b;
                    [[unroll]]for(uint h=0u;h<G;h++)part[h]=fma(qs.v[q0+h*DIM+d],kv,part[h]);
                }
            }
        }
        vec2 scp=unpackHalf2x16(sc4[g>>1u]);float sc=(g&1u)==0u?scp.x:scp.y;
        [[unroll]]for(uint h=0u;h<G;h++)s[h]=fma(part[h],sc,s[h]);
    }
#else
    uint kr=(key*p.kvheads+kh)*(DIM/8u);
    for(uint d16=0u;d16<DIM/16u;d16++){
        uvec4 ka=krow.v[kr+d16*2u],kb=krow.v[kr+d16*2u+1u];
        float k[16];
        [[unroll]]for(uint j=0u;j<4u;j++){vec2 x=unpackHalf2x16(ka[j]),y=unpackHalf2x16(kb[j]);
            k[2u*j]=x.x;k[2u*j+1u]=x.y;k[8u+2u*j]=y.x;k[8u+2u*j+1u]=y.y;}
        [[unroll]]for(uint h=0u;h<G;h++){uint qb=q0+h*DIM+d16*16u;
            [[unroll]]for(uint j=0u;j<16u;j++)s[h]=fma(qs.v[qb+j],k[j],s[h]);}
    }
#endif
    float e[4u*GQ];
    [[unroll]]for(uint h=0u;h<4u*GQ;h++)e[h]=0.0;
    [[unroll]]for(uint h=0u;h<G;h++){
        float mn=max(M[h],subgroupMax(valid?s[h]:-1e30)),r=exp(M[h]-mn);
        e[h]=valid?exp(s[h]-mn):0.0;
        L[h]=L[h]*r+e[h];M[h]=mn;o[h]*=r;
    }
    uint at=(w*64u+lane)*GQ;
    [[unroll]]for(uint q=0u;q<GQ;q++)buf[at+q]=vec4(e[4u*q],e[4u*q+1u],e[4u*q+2u],e[4u*q+3u]);
    skey[w*64u+lane]=key;
    subgroupMemoryBarrierShared();subgroupBarrier();
    // Positions past the end of the last turn have probability 0 and row 0, so the loop runs whole groups of 8.
    uint cnt=min(64u,(total-c*64u+7u)&~7u);
    for(uint j0=0u;j0<cnt;j0+=8u){
        vec4 v[8];
        [[unroll]]for(uint j=0u;j<8u;j++){uint vk=skey[w*64u+j0+j];
#if KVQ>=1
            uint vr=(vk*p.kvheads+kh)*QROW;
            int word=int(vword.v[vr+lane]);vec2 scp=unpackHalf2x16(vword.v[vr+DIM/4u+lane/16u]);
            float sc=((lane>>3u)&1u)==0u?scp.x:scp.y;
            v[j]=vec4(bitfieldExtract(word,0,8),bitfieldExtract(word,8,8),bitfieldExtract(word,16,8),
                bitfieldExtract(word,24,8))*sc;
#else
            uvec2 vv=vrow.v[(vk*p.kvheads+kh)*(DIM/4u)+lane];
            v[j]=vec4(unpackHalf2x16(vv.x),unpackHalf2x16(vv.y));
#endif
        }
        [[unroll]]for(uint j=0u;j<8u;j++){uint pa=(w*64u+j0+j)*GQ;
            vec4 pq[GQ];[[unroll]]for(uint q=0u;q<GQ;q++)pq[q]=buf[pa+q];
            [[unroll]]for(uint h=0u;h<G;h++)o[h]=fma(vec4(pq[h/4u][h%4u]),v[j],o[h]);
        }
    }
    subgroupMemoryBarrierShared();subgroupBarrier();
}
// Merge: the waves' maxima and sums, then each wave adds its rescaled accumulators in turn.
[[unroll]]for(uint h=0u;h<G;h++)L[h]=subgroupAdd(L[h]);
if(lane==0u){[[unroll]]for(uint h=0u;h<G;h++){sm[w*G+h]=M[h];sl[w*G+h]=L[h];}}
barrier();
float f[G],den[G];
[[unroll]]for(uint h=0u;h<G;h++){
    float mg=max(max(sm[h],sm[G+h]),max(sm[2u*G+h],sm[3u*G+h]));
    f[h]=exp(M[h]-mg);
    den[h]=sl[h]*exp(sm[h]-mg)+sl[G+h]*exp(sm[G+h]-mg)+sl[2u*G+h]*exp(sm[2u*G+h]-mg)+sl[3u*G+h]*exp(sm[3u*G+h]-mg);
}
for(uint k=0u;k<4u;k++){
    if(w==k){
        [[unroll]]for(uint h=0u;h<G;h++){vec4 x=o[h]*f[h];if(k>0u)x+=buf[h*64u+lane];
            if(k<3u)buf[h*64u+lane]=x;else d_v4.v[((t*p.heads+kh*G+h)*DIM)/4u+lane]=x/den[h];}
    }
    barrier();
}
}
