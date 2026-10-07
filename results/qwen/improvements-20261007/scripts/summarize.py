#!/usr/bin/env python3
"""Join timed windows to both machines' sampled clocks; flag every non-full-clock window."""
import collections, json, pathlib
root=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for path in sorted((root/'incite').glob('*.json')):
 if not path.name.startswith(('resident','final','long','corpus','cool','check')): continue
 for line in path.read_text().splitlines():
  try:d=json.loads(line)
  except ValueError:continue
  if 'pp_window_unix' not in d:continue
  row={'run':path.stem,'rep':d.get('rep'), 'case':d.get('case',{}).get('name'),'pp':d['pp'],'tg':d['tg'],'pp_tok_s':d['pp_tok_s'],'tg_tok_s':d['tg_tok_s'],'mtp':d.get('mtp'),'clocks':{}}
  for host in ['incite','invite']:
   if host=='incite':log=root/host/'clocks'/(path.stem+'.txt')
   else:
    prefix=path.stem.rsplit('-',2)[0] if path.name.startswith('resident') else path.stem
    log=root/host/'clocks'/('node-'+prefix+'.txt')
   samples=[]
   if log.exists():samples=[l.split() for l in log.read_text().splitlines()[1:]]
   row['clocks'][host]={}
   for phase in ['pp','tg']:
    a,b=d[phase+'_window_unix'];window=[s for s in samples if a<=float(s[0])<=b]
    counts=collections.Counter((s[1],s[2],s[4],s[5]) for s in window)
    row['clocks'][host][phase]={'samples':len(window),'full':sum(c for v,c in counts.items() if v==('850','1370','850','1370')),'counts':{','.join(v):c for v,c in counts.items()},'max_c':max((max(int(s[3]),int(s[6])) for s in window),default=None)}
  rows.append(row)
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
 print(row['run'],row['rep'],row['case'] or '',round(row['pp_tok_s'],2),round(row['tg_tok_s'],2),'; '.join(host+' '+','.join(phase+':'+str(v['full'])+'/'+str(v['samples']) for phase,v in data.items()) for host,data in row['clocks'].items()))
