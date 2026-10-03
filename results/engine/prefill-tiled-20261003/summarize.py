#!/usr/bin/env python3
"""Rebuild completed checks and the explicitly interrupted 80K measurement."""
from pathlib import Path
import collections,csv,hashlib,json,math,re,statistics,subprocess
r=Path(__file__).resolve().parent
project=r.parents[2]
report={}
for phase in ['baseline','candidate']:
 checked=0
 for line in (r/f'{phase}-source.sha256').read_text().splitlines():
  digest,name=line.split(None,1);name=name.strip()
  data=subprocess.check_output(['git','show','91c1672:'+name],cwd=project) if phase=='baseline' else (project/name).read_bytes()
  assert hashlib.sha256(data).hexdigest()==digest,(phase,name)
  checked+=1
 (r/f'{phase}-source-check.txt').write_text(f'PASS: {checked} files match '+('91c1672 Git content.' if phase=='baseline' else 'current source.')+'\n')
 report[f'{phase}_source_files']=checked
report['binary_hashes']={p:(r/f'{p}-binary.sha256').read_text().split()[0] for p in ['baseline','candidate']}
assert (r/'user-stop.txt').exists()
report['micro']=[]
for path in sorted(r.glob('micro-*.out')):
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 assert len(rows)==4 and all(x['pass'] and x['cpu_checks']==18 for x in rows)
 times={k:statistics.mean(x['milliseconds'] for x in rows if x['kernel']==k) for k in ['scalar','tiled']}
 report['micro'].append({'file':path.name,'scalar_ms':times['scalar'],'tiled_ms':times['tiled'],'speedup':times['scalar']/times['tiled'],'max_cpu_error':max(x['cpu_max_error'] for x in rows),'max_scalar_error':max(x['scalar_max_error'] or 0 for x in rows)})
assert len(report['micro'])==8

def progress(path):
 rows=[json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]
 assert all(math.isfinite(x['top1_score']) for x in rows)
 return rows
report['short']={}
for pp in [512,4096]:
 report['short'][str(pp)]={}
 for phase in ['baseline','candidate']:
  rows=[json.loads((r/f'{phase}-pp{pp}-{rep}.out').read_text()) for rep in [1,2]]
  report['short'][str(pp)][phase]={'pp_tok_s':sum(x['pp'] for x in rows)/sum(x['pp_seconds'] for x in rows),'tg_tok_s':sum(x['tg'] for x in rows)/sum(x['tg_seconds'] for x in rows),'runs':rows}
  for rep in [1,2]:
   a=progress(r/f'baseline-pp{pp}-{rep}.log');b=progress(r/f'{phase}-pp{pp}-{rep}.log')
   assert len(a)==len(b)==pp//512+128
   assert [(x['phase'],x['position'],x['top1_id']) for x in a]==[(x['phase'],x['position'],x['top1_id']) for x in b]
 report['short'][str(pp)]['pp_speedup']=report['short'][str(pp)]['candidate']['pp_tok_s']/report['short'][str(pp)]['baseline']['pp_tok_s']
a=progress(project/'results/engine/context-80000-20261003/ctx80000.log')
b=progress(r/'ctx80000.log')
assert 0 < len(b) < len(a) and all(x['phase']=='prefill' for x in b)
assert [(x['phase'],x['position'],x['top1_id']) for x in a[:len(b)]]==[(x['phase'],x['position'],x['top1_id']) for x in b]
last=b[-1]
matched=a[len(b)-1]
report['long']={'status':'stopped at user request','requested_pp':80000,'completed_tokens':last['position'],'completed_prefill_chunks':len(b),'elapsed_seconds':last['elapsed_seconds'],'partial_pp_tok_s':last['position']/last['elapsed_seconds'],'baseline_at_same_position_seconds':matched['elapsed_seconds'],'matched_prefix_speedup':matched['elapsed_seconds']/last['elapsed_seconds'],'decode_steps':0,'top1_ids_identical':True,'max_top1_score_difference':max(abs(x['top1_score']-y['top1_score']) for x,y in zip(a,b))}
for phase,rows in [('baseline',a[:len(b)]),('candidate',b)]:
 pre=[x for x in rows if x['phase']=='prefill']
 report['long'][phase+'_last_full_chunk']=next(x for x in reversed(pre) if x['chunk_tokens']==512)
 report['long'][phase+'_completed_bins']=[{'end':end,'tok_s':sum(x['chunk_tokens'] for x in pre if end-16000<x['position']<=end)/sum(x['chunk_seconds'] for x in pre if end-16000<x['position']<=end)} for end in range(16000,last['position']+1,16000)]
mem=list(csv.DictReader((r/'long-memory.csv').open()))
report['long']['peak_physical_mib']={k:max(float(x[k]) for x in mem)/1048576 for k in mem[0] if k!='t'}
report['checks']={}
for batch in [64,128,512]:
 row=json.loads((r/f'check-b{batch}.out').read_text());assert row['pass'] and row['serial_chunked_greedy_common_prefix']==64
 report['checks'][f'1700_b{batch}']=row
report['checks']['cpu_tests_30_passed']='test result: ok. 30 passed' in (r/'cpu-tests.log').read_text()
assert report['checks']['cpu_tests_30_passed']
assert 'PASS short suite' in (r/'short-phase.log').read_text()
report['checks']['final_acceptance']='queued suite cancelled at user request; no final reference/assistant rollback/4096 serial check on this candidate'
assert not re.search(r'GPU reset|ring.*timeout|page fault', (r/'reset-audit.txt').read_text(),re.I)
report['clocks']={}
for path in r.glob('*-clocks.txt'):
 rows=[s.split() for s in path.read_text().splitlines()[1:]]
 streak=longest=0
 for x in rows:
  streak=streak+1 if x[1:3]==['850','1370'] and x[4:6]==['850','1370'] else 0
  longest=max(longest,streak)
 report['clocks'][path.name]={'samples':len(rows),'max_temperature_c':[max(int(x[j]) for x in rows) for j in [3,6]],'clock_samples':[dict(collections.Counter(f'{x[j]}/{x[j+1]}' for x in rows)) for j in [1,4]],'longest_both_850_1370_samples':longest}
(r/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
