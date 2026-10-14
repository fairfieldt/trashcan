// QSA block selection for token t at position p: with nb = (p + 1) / kind complete blocks, keep all of them while
// nb <= p.m, else the p.m best by the scores from qwen4_idx_score (ties to the earlier block), listed in increasing
// block order. A radix select on the order-preserving bits of the scores with digits of 11, 11 and 10 bits: each
// pass counts the keys that match the digits chosen so far into a shared histogram and finds the bin holding the
// remaining rank (64 groups of 32 bins: one wave finds the group, then the bin, with inclusive scans). The selected
// blocks are then written in block order, WG at a time, their output slots counted with ballots. Up to 8 WG blocks
// the keys stay in registers between the passes.
// One workgroup per token. c: scores [t][cap / kind], d: selected blocks [t][p.m] (u32). p.kind, p.m, p.cap.
#ifndef WG
#define WG 1024u
#endif
#define WAVES (WG/64u)
layout(local_size_x=WG) in;
#define sc c_f
shared uint hist[2048];
shared uint grp[64],scan[64];
shared uint wtot[2][WAVES][2];
shared uint s_bin,s_gt,s_grp,s_acc;
uint key(float s){uint u=floatBitsToUint(s);return (u&0x80000000u)!=0u?~u:u|0x80000000u;}
void main(){
uint t=gl_WorkGroupID.y,l=gl_LocalInvocationID.x,lane=gl_SubgroupInvocationID,w=gl_SubgroupID;
if(t>=batch())return;
uint p1=position()+t+1u,nb=p1/p.kind,stride=p.cap/p.kind,top=p.m,row=t*stride;
if(nb<=top){for(uint j=l;j<nb;j+=WG)d_u.v[t*top+j]=j;return;}
bool cached=nb<=8u*WG;
uint kr[8]=uint[8](0u,0u,0u,0u,0u,0u,0u,0u);
if(cached){[[unroll]]for(uint j=0u;j<8u;j++)kr[j]=key(sc.v[row+min(j*WG+l,nb-1u)]);}
uint prefix=0u,pmask=0u,need=top;
for(uint pass=0u;pass<3u;pass++){
    uint shift=pass==0u?21u:(pass==1u?10u:0u),dmask=pass==2u?1023u:2047u;
    for(uint i=l;i<2048u;i+=WG)hist[i]=0u;
    barrier();
    if(cached){
        [[unroll]]for(uint j=0u;j<8u;j++)
            if(j*WG+l<nb&&(kr[j]&pmask)==prefix)atomicAdd(hist[(kr[j]>>shift)&dmask],1u);
    }else{
        for(uint b0=0u;b0<nb;b0+=8u*WG){
            uint k[8];
            [[unroll]]for(uint j=0u;j<8u;j++)k[j]=key(sc.v[row+min(b0+j*WG+l,nb-1u)]);
            [[unroll]]for(uint j=0u;j<8u;j++)
                if(b0+j*WG+l<nb&&(k[j]&pmask)==prefix)atomicAdd(hist[(k[j]>>shift)&dmask],1u);
        }
    }
    barrier();
    if(l<64u){uint s=0u;for(uint i=0u;i<32u;i++)s+=hist[2047u-l*32u-i];grp[l]=s;}
    barrier();
    if(w==0u){
        // The group (from the top) where the running count reaches `need`: an inclusive scan of the group counts.
        uint own=grp[lane],inc=own;
        for(uint d=1u;d<64u;d<<=1u){
            uint y=grp[lane>=d?lane-d:0u];
            subgroupMemoryBarrierShared();subgroupBarrier();
            if(lane>=d)inc+=y;
            grp[lane]=inc;
            subgroupMemoryBarrierShared();subgroupBarrier();
        }
        if(inc-own<need&&inc>=need){s_grp=lane;s_acc=inc-own;}
        subgroupMemoryBarrierShared();subgroupBarrier();
        // Then the bin within it: the same over its 32 bins.
        uint g=s_grp,bin=2047u-g*32u-(lane&31u),c=hist[bin],binc=c;
        scan[lane]=c;
        subgroupMemoryBarrierShared();subgroupBarrier();
        for(uint d=1u;d<32u;d<<=1u){
            uint y=scan[lane>=d?lane-d:0u];
            subgroupMemoryBarrierShared();subgroupBarrier();
            if((lane&31u)>=d)binc+=y;
            scan[lane]=binc;
            subgroupMemoryBarrierShared();subgroupBarrier();
        }
        uint pre=s_acc+binc-c;
        if(lane<32u&&pre<need&&pre+c>=need){s_bin=bin;s_gt=pre;}
    }
    barrier();
    prefix|=s_bin<<shift;pmask|=dmask<<shift;need-=s_gt;
    barrier();
}
// prefix is the threshold key: take every larger key and the first `need` equal ones, in block order.
uint th=prefix,ties=need,gtbase=0u,eqbase=0u;
for(uint j=0u;j*WG<nb;j++){
    uint b=j*WG+l,k=0u;
    if(cached){[[unroll]]for(uint i=0u;i<8u;i++)if(i==j)k=kr[i];}else k=key(sc.v[row+min(b,nb-1u)]);
    bool g=b<nb&&k>th,e=b<nb&&k==th;
    uvec4 bg=subgroupBallot(g),be=subgroupBallot(e);
    uint half_=j&1u;
    if(lane==0u){wtot[half_][w][0]=subgroupBallotBitCount(bg);wtot[half_][w][1]=subgroupBallotBitCount(be);}
    barrier();
    uint gw=0u,ew=0u,gall=0u,eall=0u;
    for(uint i=0u;i<WAVES;i++){uint a=wtot[half_][i][0],c=wtot[half_][i][1];
        if(i<w){gw+=a;ew+=c;}gall+=a;eall+=c;}
    uint gb=gtbase+gw+subgroupBallotExclusiveBitCount(bg),eb=eqbase+ew+subgroupBallotExclusiveBitCount(be);
    if(g||(e&&eb<ties))d_u.v[t*top+gb+min(eb,ties)]=b;
    gtbase+=gall;eqbase+=eall;
}
}
