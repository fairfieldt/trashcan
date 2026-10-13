#include "common.glsl"
#include "qwen4_expert.glsl"
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
// qwen4_moe_gu3 for batches of 2-5 tokens with each expert's rows decoded once for every token routed to it:
// workgroup (row group, item), where the items walk the batch's distinct experts in slot order, each taking up to
// CMAX of its slots (the shared expert has one per token). Every slot's result is the same bits as qwen4_moe_gu3's.
// d: ids [t][slots] (u32), at most 64 slots in the batch; otherwise as qwen4_moe_gu3 (no prefetch).
#ifndef CMAX
#define CMAX 2
#endif
#define BLOCKS 80
#define RW (BLOCKS * 4)
layout(local_size_x=256) in;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
float xscale(uvec2 hd,uint ib){
    uint ls=((hd.y>>(4u*ib))&15u)|(((hd.x>>(16u+2u*ib))&3u)<<4);
    return unpackHalf2x16(hd.x).x*float(int(ls)-32);
}
void main(){
init_lut();
uvec2 wg=(p.flags&2u)!=0?gl_WorkGroupID.yx:gl_WorkGroupID.xy;
uint lane=gl_SubgroupInvocationID,part=lane&3u,bl=lane>>2,w=gl_SubgroupID,ns=batch()*p.m;
// p.r1 = 1: only each token's last slot, expert p.r0 - 1 (the shared expert, ahead of top-k); p.r1 = 2: the others.
bool mine=lane<ns&&(p.r1==0||(p.r1==1)==(lane%p.m==p.m-1));
uint id=mine?(p.r1==1?p.r0-1:d_u.v[lane]):0xffffffffu;
// Find item wg.y: an expert's first slot k starts ceil(slots of it / CMAX) items.
uint item=wg.y;uint64_t mm=0;bool ok=false;
for(uint k=0;k<ns;k++){
    uint ek=subgroupShuffle(id,k);
    uvec4 m=subgroupBallot(id==ek);
    if(subgroupBallotFindLSB(m)!=k)continue;
    uint c=subgroupBallotBitCount(m),chunks=(c+CMAX-1)/CMAX;
    if(item<chunks){mm=packUint2x32(m.xy);for(uint s=0;s<item*CMAX;s++)mm&=mm-1ul;ok=true;break;}
    item-=chunks;
}
if(!ok)return;
uint cnt=min(uint(bitCount(unpackUint2x32(mm).x)+bitCount(unpackUint2x32(mm).y)),CMAX);
uint ts[CMAX];
[[unroll]]for(uint q=0;q<CMAX;q++){uvec2 h=unpackUint2x32(mm);ts[q]=h.x!=0?findLSB(h.x):32+findLSB(h.y);
    if(q+1<cnt)mm&=mm-1ul;}
uint e=subgroupShuffle(id,ts[0]),first=wg.x*16+w*4,uoff=p.n*RW;
uint wi=p.stride+((e*2*p.n+first)*BLOCKS+bl)*4+part;
uint sb=(e*2*p.n+first+part)*BLOCKS+bl,ub=p.n*BLOCKS,ib=bl&7u;
uint xi[CMAX];
[[unroll]]for(uint q=0;q<CMAX;q++)xi[q]=(ts[min(q,cnt-1)]/p.m)*640+bl*8+part;
vec4 g[CMAX],u[CMAX];
[[unroll]]for(uint q=0;q<CMAX;q++){g[q]=vec4(0);u[q]=vec4(0);}
[[dont_unroll]]for(uint i=0;i<BLOCKS/16;i++){
    uint cg[4],cu[4];vec4 c0[CMAX],c1[CMAX];uvec2 dg,du;
    [[unroll]]for(uint r=0;r<4;r++){cg[r]=ww.words[wi+r*RW];cu[r]=ww.words[wi+uoff+r*RW];}
    [[unroll]]for(uint q=0;q<CMAX;q++){c0[q]=b_v4.v[xi[q]];c1[q]=b_v4.v[xi[q]+4];xi[q]+=128;}
    if(FMT==2){dg=ww2.v[sb>>3];du=ww2.v[(sb+ub)>>3];}
    float sgm=FMT==2?xscale(dg,ib):unpackHalf2x16(uint(wh.halves[sb])).x;
    float sum=FMT==2?xscale(du,ib):unpackHalf2x16(uint(wh.halves[sb+ub])).x;
    wi+=64;sb+=16;
    [[unroll]]for(uint r=0;r<4;r++){
        float sg=subgroupQuadBroadcast(sgm,r),su=subgroupQuadBroadcast(sum,r);
        vec4 gl=qlo(cg[r]),gh=qhi(cg[r]),ul=qlo(cu[r]),uh=qhi(cu[r]);
        [[unroll]]for(uint q=0;q<CMAX;q++){
            g[q][r]=fma(dot(gl,c0[q])+dot(gh,c1[q]),sg,g[q][r]);
            u[q][r]=fma(dot(ul,c0[q])+dot(uh,c1[q]),su,u[q][r]);}
    }
}
[[unroll]]for(uint q=0;q<CMAX;q++)if(q<cnt){vec4 a=subgroupAdd(g[q]),c=subgroupAdd(u[q]);
    if(lane==0)[[unroll]]for(uint r=0;r<4;r++)if(first+r<p.n)c_f.v[ts[q]*p.n+first+r]=a[r]/(1.0+exp(-a[r]))*c[r];}
}
