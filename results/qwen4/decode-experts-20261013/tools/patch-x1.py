# Experimental build only (trashcan-moe-x): top-k timing probes (p.flags) and a top-k variant with the expert count
# per lane a compile-time constant and branch-free compaction (qwen4_topk3). Reverted after building.
wt='/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/'
p=wt+'engine/shaders/qwen4_topk.comp'
s=open(p).read()
s=s.replace('''float gate=a_f.v[row+p.n];
''','''float gate=a_f.v[row+p.n];
if((p.flags&1u)!=0){float z=gate;[[unroll]]for(uint j=0;j<8;j++)z+=v[j];if(l==0)d_f.v[t*slots]=z;return;}
''',1)
s=s.replace('''mx=subgroupMax(mx);float sum=0;''','''mx=subgroupMax(mx);float sum=0;
if((p.flags&2u)!=0){if(l==0)d_f.v[t*slots]=mx;return;}''',1)
s=s.replace('''uint u[8],mk=0;''','''if((p.flags&4u)!=0){if(l==0)d_f.v[t*slots]=v[0];return;}
uint u[8],mk=0;''',1)
s=s.replace('''uint cnt=0;''','''if((p.flags&8u)!=0){if(l==0)c_u.v[t*slots]=thr;return;}
uint cnt=0;''',1)
open(p,'w').write(s)
# topk3: PER = 8 at compile time, compaction without branches (dummy slots 64..127 for lanes not picked).
t3=s.replace('''uint row=t*(p.n+1),per=p.n/64,slots=p.m+1,K=p.m;''','''uint row=t*(p.n+1),slots=p.m+1,K=p.m;const uint per=8;''')
t3=t3.replace('''shared uint sid[64],sk[64];''','''shared uint sid[128],sk[128];''')
t3=t3.replace('''    if(s){uint k=pos+subgroupBallotExclusiveBitCount(b);sid[k]=j*64+l;sk[k]=u[j];}''','''    uint k=s?pos+subgroupBallotExclusiveBitCount(b):64+l;sid[k]=j*64+l;sk[k]=u[j];''')
open(wt+'engine/shaders/qwen4_topk3.comp','w').write(t3)
p=wt+'engine/build.rs'
s=open(p).read()
s=s.replace('        "qwen4_router_gemm",\n','        "qwen4_router_gemm",\n        "qwen4_topk3",\n',1)
open(p,'w').write(s)
p=wt+'engine/src/vulkan.rs'
s=open(p).read()
old='            ("qwen4_router_gemm", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_router_gemm.spv")).as_slice()),\n'
s=s.replace(old,old+'            ("qwen4_topk3", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_topk3.spv")).as_slice()),\n',1)
open(p,'w').write(s)
p=wt+'engine/src/qwen4/gpu.rs'
s=open(p).read()
old='''            gpu.dispatch(
                current.get(),
                "qwen4_topk",'''
new='''            let xab = std::env::var("TRASHCAN_MOE_X").unwrap_or_default();
            u[11] = xab.strip_prefix("tkp").and_then(|v| v.parse().ok()).unwrap_or(0);
            gpu.dispatch(
                current.get(),
                if xab == "tk3" { "qwen4_topk3" } else { "qwen4_topk" },'''
assert old in s; s=s.replace(old,new,1)
open(p,'w').write(s)
print('ok')
