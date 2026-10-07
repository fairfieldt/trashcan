// QSA attention over the positions a token keeps: the selected complete blocks of p.kind positions (in block order)
// then the incomplete tail ending at the token. Otherwise as attention.comp / attention_partial.comp: one wave
// per (head, token[, partition]), online softmax, K/V caches of p.cap rows: f16 [cap][kvheads][dim], or with KVQ 1
// (K and V) or 2 (V only) Q8 rows of dim/4 words of int8 values then dim/32 f16 scales (qwen4_kv_store).
// a: q (pre-scaled) [t][heads][dim], b: K cache, c: V cache, d: output [t][heads][dim], e: partition scratch,
// f: selected blocks [t][p.m] (u32). p.heads, p.kvheads, p.dim, p.cap, p.kind = block size, p.m = block budget.
layout(local_size_x=64) in;
#ifndef KVQ
#define KVQ 0
#endif
float q8_b(uint row,uint d){int q=bitfieldExtract(int(b_u.v[row+(d>>2)]),int((d&3u)*8u),8);
    vec2 s=unpackHalf2x16(b_u.v[row+p.dim/4u+(d>>6)]);return float(q)*(((d>>5)&1u)==0u?s.x:s.y);}
float q8_c(uint row,uint d){int q=bitfieldExtract(int(c_u.v[row+(d>>2)]),int((d&3u)*8u),8);
    vec2 s=unpackHalf2x16(c_u.v[row+p.dim/4u+(d>>6)]);return float(q)*(((d>>5)&1u)==0u?s.x:s.y);}
#ifdef PARTIAL
layout(constant_id=0) const uint parts=32;
#endif
void main(){
#ifdef PARTIAL
uint chunk=gl_WorkGroupID.x%parts,head=gl_WorkGroupID.x/parts;
#else
uint head=gl_WorkGroupID.x;
#endif
uint t=gl_WorkGroupID.y,l=gl_LocalInvocationID.x;
if(t>=batch())return;
uint pos=position()+t,nb=(pos+1)/p.kind,nsel=min(nb,p.m),selpos=nsel*p.kind,total=selpos+pos+1-nb*p.kind;
uint start=0,end=total;
#ifdef PARTIAL
uint step=(total+parts-1)/parts;start=min(total,chunk*step);end=min(total,start+step);
#endif
uint kh=head/(p.heads/p.kvheads),qbase=t*p.heads*p.dim+head*p.dim;
float acc[8],q[8];[[unroll]] for(uint j=0;j<8;j++){acc[j]=0;q[j]=j*64+l<p.dim?a_f.v[qbase+j*64+l]:0;}
float maxs=-1e30,den=0;
for(uint i=start;i<end;i++){
    uint key=i<selpos?f_u.v[t*p.m+i/p.kind]*p.kind+i%p.kind:nb*p.kind+(i-selpos);
    uint kb=(key%p.cap)*p.kvheads*p.dim+kh*p.dim,qr=((key%p.cap)*p.kvheads+kh)*(p.dim/4u+p.dim/64u);
    float dot=0;
#if KVQ==1
    [[unroll]] for(uint j=0;j<p.dim/64;j++)dot=fma(q[j],q8_b(qr,j*64+l),dot);
#else
    [[unroll]] for(uint j=0;j<p.dim/64;j++)dot=fma(q[j],half_b(kb+j*64+l),dot);
#endif
    float score=subgroupAdd(dot),nm=max(maxs,score),rescale=exp(maxs-nm),w=exp(score-nm);
    den=den*rescale+w;
#if KVQ>=1
    [[unroll]] for(uint j=0;j<p.dim/64;j++)acc[j]=fma(w,q8_c(qr,j*64+l),acc[j]*rescale);
#else
    [[unroll]] for(uint j=0;j<p.dim/64;j++)acc[j]=fma(w,half_c(kb+j*64+l),acc[j]*rescale);
#endif
    maxs=nm;
}
#ifdef PARTIAL
uint base=((t*p.heads+head)*parts+chunk)*(p.dim+2);
[[unroll]] for(uint j=0;j<p.dim/64;j++)e_f.v[base+2+j*64+l]=acc[j];
if(l==0){e_f.v[base]=maxs;e_f.v[base+1]=den;}
#else
[[unroll]] for(uint j=0;j<p.dim/64;j++)d_f.v[qbase+j*64+l]=acc[j]/den;
#endif
}
