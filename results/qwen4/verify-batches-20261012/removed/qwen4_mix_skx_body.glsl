#include "common.glsl"
// EXPERIMENT (verify-batches): qwen4_mix_down_sk_body (Q8_0 only) with ROWS rows per wave (define), the column count
// exact (1-4), with PF the next iteration's weights, scales and inputs loaded before the current one is computed,
// with NOFIN (diagnostic, wrong results) no finishing pass, chunks of CB blocks (40, 16 or 80 of a stream's 80) and
// with W2 four lanes per block loading 8 weights each (else 8 lanes, 4 weights).
// Specialization constant: COLS | PF << 3 | NOFIN << 4 | CBSEL << 5 (0: 40, 1: 16, 2: 80) | W2 << 7.
layout(constant_id=0) const uint SPEC=1;
const uint COLS=SPEC&7u;
const bool PF=(SPEC&8u)!=0u;
const bool NOFIN=(SPEC&16u)!=0u;
const uint CBSEL=(SPEC>>5)&3u;
const uint CB=CBSEL==1u?16u:(CBSEL==2u?80u:40u);
const bool W2=(SPEC&128u)!=0u;
const uint CPS=80u/CB;
layout(set=2,binding=2,std430) coherent buffer Partials { float v[]; } cp;
layout(set=2,binding=7,std430) coherent buffer Counters { uint v[]; } cnt;
layout(binding=0,std430) readonly restrict buffer WeightU2 { uvec2 v[]; } ww2;
layout(local_size_x=256) in;
shared bool last;
vec4 i8(uint w){return vec4(ivec4(w<<24,w<<16,w<<8,w)>>24);}
void main(){
uint lane=gl_SubgroupInvocationID,blocks=p.k/32,sb=p.n/32,C=blocks/CB,nb=batch();
uint t0=(gl_WorkGroupID.y/C)*COLS,c=gl_WorkGroupID.y%C;
uint rows=p.m+4,first=gl_WorkGroupID.x*(4*ROWS)+gl_SubgroupID*ROWS,s=c*CB/sb,P=p.m+8;
bool comb=(p.flags&1)!=0;
float ws[4],ss[4],acc[ROWS][4];
uint tt[4];
[[unroll]]for(uint z=0;z<4;z++){tt[z]=min(t0+z,nb-1);ws[z]=comb?d_f.v[tt[z]*P+p.m+s]:0.0;ss[z]=0;
    [[unroll]]for(uint r=0;r<ROWS;r++)acc[r][z]=0;}
uint rb[ROWS];[[unroll]]for(uint r=0;r<ROWS;r++)rb[r]=min(first+r,rows-1)*blocks;
if(W2){uint part=lane%4;
    for(uint b=c*CB+lane/4;b<(c+1)*CB;b+=16){
        uint i0=(b*32-s*p.n)/4+part*2;
        uvec2 q[ROWS];uint d[ROWS];
        [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;q[r]=ww2.v[p.stride/2+blk*4+part];d[r]=uint(wh.halves[blk]);}
        vec4 e0=e_v4.v[s*p.n/4+i0],e1=e_v4.v[s*p.n/4+i0+1];
        [[unroll]]for(uint z=0;z<COLS;z++){uint xo=(tt[z]*p.k+b*32)/4+part*2;
            vec4 x0=a_v4.v[xo],x1=a_v4.v[xo+1];
            if(comb){x0=fma(vec4(ws[z]),b_v4.v[tt[z]*p.n/4+i0],x0);x1=fma(vec4(ws[z]),b_v4.v[tt[z]*p.n/4+i0+1],x1);}
            ss[z]+=dot(x0,x0)+dot(x1,x1);
            vec4 g0=x0*e0,g1=x1*e1;
            [[unroll]]for(uint r=0;r<ROWS;r++)acc[r][z]=fma(dot(i8(q[r].x),g0)+dot(i8(q[r].y),g1),unpackHalf2x16(d[r]).x,acc[r][z]);}
    }
}else{
uint part=lane%8;
uint qn[ROWS],dn[ROWS];vec4 en,xn[4],yn[4];
uint b=c*CB+lane/8;
if(PF){uint i0=(b*32-s*p.n)/4+part;
    [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;qn[r]=ww.words[p.stride+blk*8+part];dn[r]=uint(wh.halves[blk]);}
    en=e_v4.v[s*p.n/4+i0];
    [[unroll]]for(uint z=0;z<COLS;z++){xn[z]=a_v4.v[(tt[z]*p.k+b*32)/4+part];yn[z]=b_v4.v[tt[z]*p.n/4+i0];}}
for(;b<(c+1)*CB;b+=8){
    uint q[ROWS],d[ROWS];vec4 e0,x[4],y[4];
    uint i0=(b*32-s*p.n)/4+part;
    if(PF){q=qn;d=dn;e0=en;x=xn;y=yn;uint bn=min(b+8,(c+1)*CB-1),i1=(bn*32-s*p.n)/4+part;
        [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+bn;qn[r]=ww.words[p.stride+blk*8+part];dn[r]=uint(wh.halves[blk]);}
        en=e_v4.v[s*p.n/4+i1];
        [[unroll]]for(uint z=0;z<COLS;z++){xn[z]=a_v4.v[(tt[z]*p.k+bn*32)/4+part];yn[z]=b_v4.v[tt[z]*p.n/4+i1];}
    }else{
        [[unroll]]for(uint r=0;r<ROWS;r++){uint blk=rb[r]+b;q[r]=ww.words[p.stride+blk*8+part];d[r]=uint(wh.halves[blk]);}
        e0=e_v4.v[s*p.n/4+i0];
        [[unroll]]for(uint z=0;z<COLS;z++){x[z]=a_v4.v[(tt[z]*p.k+b*32)/4+part];y[z]=b_v4.v[tt[z]*p.n/4+i0];}
    }
    [[unroll]]for(uint z=0;z<COLS;z++){vec4 x0=x[z];
        if(comb)x0=fma(vec4(ws[z]),y[z],x0);
        ss[z]+=dot(x0,x0);
        vec4 g0=x0*e0;
        [[unroll]]for(uint r=0;r<ROWS;r++)acc[r][z]=fma(dot(i8(q[r]),g0),unpackHalf2x16(d[r]).x,acc[r][z]);}
}
}
[[unroll]]for(uint z=0;z<COLS;z++)if(t0+z<nb){uint o=((t0+z)*C+c)*P;
    [[unroll]]for(uint r=0;r<ROWS;r++){float v=subgroupAdd(acc[r][z]);if(lane==0&&first+r<rows)cp.v[o+first+r]=v;}
    if(gl_WorkGroupID.x==0&&gl_SubgroupID==0){float v=subgroupAdd(ss[z]);if(lane==0)cp.v[o+rows]=v;}}
if(NOFIN)return;
memoryBarrierBuffer();
barrier();
uint tg=gl_WorkGroupID.y/C,tid=gl_LocalInvocationID.x;
if(tid==0)last=atomicAdd(cnt.v[tg],1u)==gl_NumWorkGroups.x*C-1u;
barrier();
if(!last)return;
memoryBarrierBuffer();
if(tid==0)cnt.v[tg]=0;
float eps=uintBitsToFloat(p.r0);
for(uint z=0;z<COLS;z++)if(t0+z<nb){uint t=t0+z,base=t*C*P,o=t*P;
    float q[4][CPS],a[2][4][CPS];
    [[unroll]]for(uint s=0;s<4;s++)[[unroll]]for(uint i=0;i<CPS;i++){uint at=base+(s*CPS+i)*P;q[s][i]=cp.v[at+p.m+4];
        [[unroll]]for(uint h=0;h<2;h++)a[h][s][i]=cp.v[at+min(tid+256*h,rows-1)];}
    float inv[4];
    [[unroll]]for(uint s=0;s<4;s++){float qs=0;[[unroll]]for(uint i=0;i<CPS;i++)qs+=q[s][i];
        inv[s]=inversesqrt(qs/float(p.n)+eps);}
    [[unroll]]for(uint h=0;h<2;h++){uint j=tid+256*h;if(j<rows){float v=0;
        [[unroll]]for(uint s=0;s<4;s++){float as=0;[[unroll]]for(uint i=0;i<CPS;i++)as+=a[h][s][i];v+=inv[s]*as;}
        float y=v*0.25;f_f.v[o+j]=j<p.m?y/(1.0+exp(-y)):2.0/(1.0+exp(-y));}}
    if(tid<4)f_f.v[o+p.m+4+tid]=inv[tid];}
}
