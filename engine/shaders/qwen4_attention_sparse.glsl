// QSA attention over the positions a token keeps: the selected complete blocks of p.kind positions (in block order)
// then the incomplete tail ending at the token. Otherwise as attention.comp / attention_partial.comp: one wave
// per (head, token[, partition]), online softmax, f16 K/V caches of p.cap rows.
// a: q (pre-scaled) [t][heads][dim], b: K cache, c: V cache, d: output [t][heads][dim], e: partition scratch,
// f: selected blocks [t][p.m] (u32). p.heads, p.kvheads, p.dim, p.cap, p.kind = block size, p.m = block budget.
layout(local_size_x=64) in;
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
    uint kb=(key%p.cap)*p.kvheads*p.dim+kh*p.dim;
    float dot=0;
    [[unroll]] for(uint j=0;j<p.dim/64;j++)dot=fma(q[j],half_b(kb+j*64+l),dot);
    float score=subgroupAdd(dot),nm=max(maxs,score),rescale=exp(maxs-nm),w=exp(score-nm);
    den=den*rescale+w;
    [[unroll]] for(uint j=0;j<p.dim/64;j++)acc[j]=fma(w,half_c(kb+j*64+l),acc[j]*rescale);
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
