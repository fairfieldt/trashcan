#include "common.glsl"
layout(local_size_x=HEADS_PER_GROUP*64) in;
layout(constant_id=0) const uint parts=32;
shared float tile_k[SHARED_FLOATS],tile_v[SHARED_FLOATS];
void main(){
    uint chunk=gl_WorkGroupID.x%parts;
    uint first_head=(gl_WorkGroupID.x/parts)*HEADS_PER_GROUP;
    uint t=gl_WorkGroupID.y,l=gl_SubgroupInvocationID;
    // Selected callers have eight query heads and one KV plane per group.
    // Preserve the uniform dispatch guard; other head mappings are unsupported.
    if(first_head>=p.heads||t>=batch())return;
    uint head=first_head+gl_SubgroupID;
    bool valid_head=head<p.heads;
    uint queries_per_kv=p.heads/p.kvheads;
    uint first_kv=first_head/queries_per_kv;
    uint last_kv=(min(first_head+HEADS_PER_GROUP,p.heads)-1)/queries_per_kv;
    uint kv_count=last_kv-first_kv+1;
    uint local_kv=min(head,p.heads-1)/queries_per_kv-first_kv;
    uint shared_base=local_kv*DIM;
    uint pos=position()+t,start=p.window>0 && pos+1>p.window ? pos+1-p.window:0;
    uint end=params_u.v[2]!=0 ? min(pos,params_u.v[2]-1):pos;
    uint step=(end-start+parts)/parts;
    uint lo=start+chunk*step;end=min(end,lo+step-1);start=lo;
    uint qbase=t*p.heads*DIM+head*DIM;
    float acc[DIM/64],q[DIM/64];
    [[unroll]]for(uint j=0;j<DIM/64;j++){
        acc[j]=0;
        q[j]=valid_head&&j*64+l<DIM?a_f.v[qbase+j*64+l]:0;
    }
    float maxs=-1e30,den=0;
    for(uint key=start;key<=end;key++){
        // Each distinct KV head is loaded once for this query-head group.
        // Selected eight-query-head shapes use one KV plane per group.
        uint kb=(key%p.cap)*p.kvheads*DIM+first_kv*DIM;
        for(uint i=gl_LocalInvocationID.x;i<kv_count*DIM;i+=gl_WorkGroupSize.x){
            tile_k[i]=half_b(kb+i);
            tile_v[i]=half_c(kb+i);
        }
        barrier();
        float dot=0;
        [[unroll]]for(uint j=0;j<DIM/64;j++)
            dot=fma(q[j],tile_k[shared_base+j*64+l],dot);
        float score=subgroupAdd(dot),nm=max(maxs,score),rescale=exp(maxs-nm),w=exp(score-nm);
        den=den*rescale+w;
        [[unroll]]for(uint j=0;j<DIM/64;j++)
            acc[j]=fma(w,tile_v[shared_base+j*64+l],acc[j]*rescale);
        maxs=nm;
        // All head waves must finish consuming this key before its LDS slots
        // can be overwritten by the next key, including the final iteration.
        barrier();
    }
    if(valid_head){
        uint base=((t*p.heads+head)*parts+chunk)*(DIM+2);
        [[unroll]]for(uint j=0;j<DIM/64;j++)e_f.v[base+2+j*64+l]=acc[j];
        if(l==0){e_f.v[base]=maxs;e_f.v[base+1]=den;}
    }
}
