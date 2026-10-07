#!/usr/bin/env python3
"""Summarize paired prefill runs and match their timing windows to sampled GPU clocks."""
import collections, json, pathlib, statistics
root=pathlib.Path(__file__).resolve().parents[1]
rows=[]
reference=None
for path in sorted((root/'incite').glob('slab-*.json')):
 for rep,line in enumerate(path.read_text().splitlines()):
  d=json.loads(line)
  if 'pp_window_unix' not in d:continue
  check=[d['pp_top'],d['token_ids']]
  if reference is None:reference=check
  assert check==reference,(path.name,rep,'candidate scores or generated IDs differ')
  row={'run':path.stem,'rep':rep,'slab':d['experimental']['attention_slab'],'pp_tok_s':d['pp_tok_s'],'device_mib':d['local_device_mib'],'clocks':{}}
  for host in ['incite','invite']:
   name=('node-' if host=='invite' else '')+path.stem+'.txt'
   samples=[s.split() for s in (root/host/'clocks'/name).read_text().splitlines()[1:]]
   a,b=d['pp_window_unix'];samples=[s for s in samples if a<=float(s[0])<=b]
   counts=collections.Counter(','.join([s[1],s[2],s[4],s[5]]) for s in samples)
   row['clocks'][host]={'samples':len(samples),'full':counts['850,1370,850,1370'],'counts':dict(counts),'max_c':max(max(int(s[3]),int(s[6])) for s in samples)}
  rows.append(row)
(root/'summary.json').write_text(json.dumps({'exact_candidates_and_ids':True,'rows':rows},indent=2)+'\n')
for row in rows:print(row['run'],row['rep'],round(row['pp_tok_s'],2),row['clocks'])
for slab in [64,256]:
 values=[r['pp_tok_s'] for r in rows if r['slab']==slab]
 if values:print('slab',slab,'median',statistics.median(values),'range',min(values),max(values))
