// QSA attention for decode steps and verification batches (up to 8 tokens), grouped by KV head: a workgroup per
// (KV head, partition of 64 visible positions, token) writes partial results for the G = 12 query heads that share
// the KV head, in attention_merge's layout ([max, sum, unnormalized output] per head and partition; only the
// partitions holding the token's positions, which qwen4_attention_merge reads). A lane takes a position: each wave
// scores a quarter of its K row (64 dims) against the 12 queries (wave-uniform, scalar loads), the waves' partial
// sums meet in shared memory, each wave turns 3 heads' scores into probabilities, then each wave adds 16 positions'
// V rows times all 12 heads' probabilities (4 dims per lane), and the four waves' sums merge in shared memory.
// The loops over heads, positions and merge rounds stay loops: a decode step runs this code once per wave, and on
// GCN1 every dispatch fetches its instructions anew (about 7 us per KB of code executed once; fully unrolled, the
// kernel was 12.5 KB and took 169 us at 2049 positions, 137 us at 129).
// a: q (pre-scaled) [t][heads][dim], b: K cache, c: V cache, e: partials [t][heads][parts][dim + 2], f: selected
// blocks [t][p.m]. K/V caches as in qwen4_attention_grouped.glsl (KVQ). p.heads, p.kvheads, p.dim = 256, p.cap,
// p.kind, p.m; p.heads / p.kvheads = 12.
layout(local_size_x=256) in;
layout(constant_id=0) const uint parts=64;
#ifndef KVQ
#define KVQ 0
#endif
#define G 12u
#define DIM 256u
#define QROW (DIM/4u+DIM/64u)
layout(set=2,binding=0,std430) readonly restrict buffer Queries { vec4 v[]; } qs4;
layout(set=2,binding=1,std430) readonly restrict buffer KRows { uvec4 v[]; } krow;
layout(set=2,binding=2,std430) readonly restrict buffer VRows { uvec2 v[]; } vrow;
layout(set=2,binding=2,std430) readonly restrict buffer VWords { uint v[]; } vword;
layout(set=2,binding=5,std430) readonly restrict buffer Selected { uint v[]; } selb;
// The partial scores [wave][head][position] (as floats), later the merged output of the waves ([head][64] vec4).
shared vec4 buf[G*64u];
#define PART(i) buf[(i)>>2u][(i)&3u]
// The probabilities [position][head quad].
shared vec4 prob[64u*3u];
shared uint skey[64];
shared float sm[G],sl[G];

void main(){
uint kh=gl_WorkGroupID.x/parts,chunk=gl_WorkGroupID.x%parts,t=gl_WorkGroupID.y;
uint l=gl_LocalInvocationID.x,lane=gl_SubgroupInvocationID,w=gl_SubgroupID;
if(t>=batch())return;
uint pos=position()+t,nb=(pos+1u)/p.kind,nsel=min(nb,p.m),selpos=nsel*p.kind,total=selpos+pos+1u-nb*p.kind;
uint i0=chunk*64u;
if(i0>=total)return;
uint cnt=min(64u,total-i0),h0=t*p.heads+kh*G;
uint i=i0+lane;bool valid=lane<cnt;
uint key=i<selpos?selb.v[t*p.m+i/p.kind]*p.kind+i%p.kind:nb*p.kind+(i-selpos);
key=valid?key%p.cap:0u;
if(w==0u)skey[lane]=key;
barrier();
// Partial scores over dims 64w .. 64w+63: this quarter of the K row as 16 vec4s, then each head's 16 query vec4s.
vec4 kq[16];
#if KVQ==1
uint kr=(key*p.kvheads+kh)*(QROW/4u);
vec2 scp=unpackHalf2x16(krow.v[kr+DIM/16u][w]);
[[unroll]]for(uint u=0u;u<4u;u++){
    uvec4 kk=krow.v[kr+w*4u+u];float sc=u<2u?scp.x:scp.y;
    [[unroll]]for(uint j=0u;j<4u;j++){int word=int(kk[j]);
        kq[u*4u+j]=vec4(bitfieldExtract(word,0,8),bitfieldExtract(word,8,8),bitfieldExtract(word,16,8),
            bitfieldExtract(word,24,8))*sc;}
}
#else
uint kr=(key*p.kvheads+kh)*(DIM/8u)+w*8u;
[[unroll]]for(uint u=0u;u<8u;u++){
    uvec4 kk=krow.v[kr+u];
    kq[2u*u]=vec4(unpackHalf2x16(kk.x),unpackHalf2x16(kk.y));kq[2u*u+1u]=vec4(unpackHalf2x16(kk.z),unpackHalf2x16(kk.w));
}
#endif
[[dont_unroll]]for(uint h=0u;h<G;h++){
    float acc=0.0;
    [[unroll]]for(uint j=0u;j<16u;j++)acc+=dot(qs4.v[(h0+h)*64u+w*16u+j],kq[j]);
    PART((w*G+h)*64u+lane)=acc;
}
barrier();
// Heads 3w .. 3w+2: the four quarters' sum, then this partition's maximum, probabilities and sum.
[[dont_unroll]]for(uint hh=0u;hh<3u;hh++){uint h=3u*w+hh;
    float sh=PART(h*64u+lane)+PART((G+h)*64u+lane)+PART((2u*G+h)*64u+lane)+PART((3u*G+h)*64u+lane);
    float mx=subgroupMax(valid?sh:-1e30);
    float e=valid?exp(sh-mx):0.0;
    float sum=subgroupAdd(e);
    if(lane==0u){sm[h]=mx;sl[h]=sum;}
    prob[lane*3u+h/4u][h%4u]=e;
}
barrier();
// Values: wave w adds positions 16w .. 16w+15 for all heads, four positions' V rows loaded together (one wait for
// memory per four positions instead of one per position), in position order.
vec4 o[G];[[unroll]]for(uint h=0u;h<G;h++)o[h]=vec4(0.0);
uint j1=min(16u*w+16u,cnt);
[[dont_unroll]]for(uint j=16u*w;j<j1;j+=4u){
    vec4 v[4];
    [[unroll]]for(uint q=0u;q<4u;q++){uint vk=skey[min(j+q,63u)];
#if KVQ>=1
        uint vr=(vk*p.kvheads+kh)*QROW;
        int word=int(vword.v[vr+lane]);vec2 vsc=unpackHalf2x16(vword.v[vr+DIM/4u+lane/16u]);
        float sc=((lane>>3u)&1u)==0u?vsc.x:vsc.y;
        v[q]=vec4(bitfieldExtract(word,0,8),bitfieldExtract(word,8,8),bitfieldExtract(word,16,8),
            bitfieldExtract(word,24,8))*sc;
#else
        uvec2 vv=vrow.v[(vk*p.kvheads+kh)*(DIM/4u)+lane];
        v[q]=vec4(unpackHalf2x16(vv.x),unpackHalf2x16(vv.y));
#endif
    }
    [[unroll]]for(uint q=0u;q<4u;q++)if(j+q<j1){uint jq=j+q;
        vec4 p0=prob[jq*3u],p1=prob[jq*3u+1u],p2=prob[jq*3u+2u];
        o[0]=fma(vec4(p0.x),v[q],o[0]);o[1]=fma(vec4(p0.y),v[q],o[1]);o[2]=fma(vec4(p0.z),v[q],o[2]);
        o[3]=fma(vec4(p0.w),v[q],o[3]);o[4]=fma(vec4(p1.x),v[q],o[4]);o[5]=fma(vec4(p1.y),v[q],o[5]);
        o[6]=fma(vec4(p1.z),v[q],o[6]);o[7]=fma(vec4(p1.w),v[q],o[7]);o[8]=fma(vec4(p2.x),v[q],o[8]);
        o[9]=fma(vec4(p2.y),v[q],o[9]);o[10]=fma(vec4(p2.z),v[q],o[10]);o[11]=fma(vec4(p2.w),v[q],o[11]);}
}
// The partial scores are dead (the barrier after the probabilities): their table holds the waves' running sums.
[[dont_unroll]]for(uint k=0u;k<4u;k++){
    if(w==k){
        if(k>0u){[[unroll]]for(uint h=0u;h<G;h++)o[h]+=buf[h*64u+lane];}
        if(k<3u){[[unroll]]for(uint h=0u;h<G;h++)buf[h*64u+lane]=o[h];}
        else{
            [[unroll]]for(uint h=0u;h<G;h++){uint b=((h0+h)*parts+chunk)*(DIM+2u);vec4 x=o[h];
                e_f.v[b+2u+4u*lane]=x.x;e_f.v[b+3u+4u*lane]=x.y;e_f.v[b+4u+4u*lane]=x.z;e_f.v[b+5u+4u*lane]=x.w;}
            if(lane<G){uint b=((h0+lane)*parts+chunk)*(DIM+2u);e_f.v[b]=sm[lane];e_f.v[b+1u]=sl[lane];}
        }
    }
    barrier();
}
}
