# Rebuild the session-5 top-k variant (qwen4_topk3, with the probes of the x build) from the current qwen4_topk.comp
# into the results' removed/ directory, without touching the engine sources.
wt = '/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/'
s = open(wt + 'engine/shaders/qwen4_topk.comp').read()
reps = [
    ('''float gate=a_f.v[row+p.n];
''', '''float gate=a_f.v[row+p.n];
if((p.flags&1u)!=0){float z=gate;[[unroll]]for(uint j=0;j<8;j++)z+=v[j];if(l==0)d_f.v[t*slots]=z;return;}
'''),
    ('''mx=subgroupMax(mx);float sum=0;''', '''mx=subgroupMax(mx);float sum=0;
if((p.flags&2u)!=0){if(l==0)d_f.v[t*slots]=mx;return;}'''),
    ('''uint u[8],mk=0;''', '''if((p.flags&4u)!=0){if(l==0)d_f.v[t*slots]=v[0];return;}
uint u[8],mk=0;'''),
    ('''uint cnt=0;''', '''if((p.flags&8u)!=0){if(l==0)c_u.v[t*slots]=thr;return;}
uint cnt=0;'''),
    ('''uint row=t*(p.n+1),per=p.n/64,slots=p.m+1,K=p.m;''', '''uint row=t*(p.n+1),slots=p.m+1,K=p.m;const uint per=8;'''),
    ('''shared uint sid[64],sk[64];''', '''shared uint sid[128],sk[128];'''),
    ('''    if(s){uint k=pos+subgroupBallotExclusiveBitCount(b);sid[k]=j*64+l;sk[k]=u[j];}''',
     '''    uint k=s?pos+subgroupBallotExclusiveBitCount(b):64+l;sid[k]=j*64+l;sk[k]=u[j];'''),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
hdr = ('// Session 5 (removed): qwen4_topk with 8 keys per lane at compile time and compaction without branches (lanes\n'
       '// not picked write to slots 64..127), plus the timing probes of the x build (p.flags 1/2/4/8: return after the\n'
       '// loads, the maximum, the probabilities, the threshold). 28.7 against 31.6 us at 1 token, 28.8 against 29.9 at 3.\n')
open(wt + 'results/qwen4/decode-experts-20261013/removed/qwen4_topk3.comp', 'w').write(hdr + s)
print('ok')
