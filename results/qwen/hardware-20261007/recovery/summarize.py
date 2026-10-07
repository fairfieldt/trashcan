#!/usr/bin/env python3
"""Summarize recovered data without modifying raw evidence."""
import collections, datetime, json, pathlib, statistics, tarfile
root=pathlib.Path(__file__).resolve().parent
old={'incite':'c7692794-b9bd-44fe-bfe8-2401ddd8f04d','invite':'17105e70-09ac-496f-bda1-6d03e2a72fca'}
report={'slab':[],'recorders':{},'governors':{}}
reference=None
for path in sorted((root/'incite/interrupted-workspace/results').glob('slab-*.json')):
 for line in path.read_text().splitlines():
  if not line.strip():continue
  d=json.loads(line);check=[d['pp_top'],d['token_ids']]
  if reference is None:reference=check
  row={'run':path.stem,'rep':d['rep'],'slab':d['experimental']['attention_slab'],'pp_tok_s':d['pp_tok_s'],'tg_tok_s':d['tg_tok_s'],'exact_candidates_and_ids':check==reference,'clocks':{}}
  for h in old:
   p=root/h/'interrupted-workspace/results/clocks'/((('node-' if h=='invite' else '')+path.stem+'.txt'))
   raw=p.read_bytes();samples=[];invalid=0
   for sample in raw.decode().splitlines()[1:]:
    cells=sample.split()
    try:
     if len(cells)!=7:raise ValueError()
     values=list(map(float,cells))
    except ValueError:invalid+=1;continue
    if d['pp_window_unix'][0]<=values[0]<=d['pp_window_unix'][1]:samples.append(values)
   counts=collections.Counter(','.join(str(int(x[i])) for i in [1,2,4,5]) for x in samples)
   row['clocks'][h]={'samples':len(samples),'full':counts['850,1370,850,1370'],'counts':dict(counts),'max_c':max([max(s[3],s[6]) for s in samples],default=None),'file_nul_bytes':raw.count(b'\0'),'file_invalid_lines':invalid}
  report['slab'].append(row)
for h,boot in old.items():
 records=[];bad=0
 with tarfile.open(root/h/'recorders.tar.gz') as archive:
  text_files={m.name:archive.extractfile(m).read().decode() for m in archive.getmembers() if m.isfile() and (m.name.startswith('trashcan-crash/telemetry.jsonl') or m.name.startswith('trashcan-platform/events.jsonl'))}
  dumps=[m.name for m in archive.getmembers() if m.name.endswith('.dump')]
 for name,contents in text_files.items():
  if not name.startswith('trashcan-crash/telemetry.jsonl'):continue
  for line in contents.splitlines():
   try:d=json.loads(line)
   except ValueError:bad+=1;continue
   if d.get('boot')==boot:records.append(d)
 records.sort(key=lambda x:x['utc'])
 latest=records[-1] if records else None
 ending=[r for r in records if r['utc']>='2026-10-07T19:09:00']
 (root/h/'prior-telemetry-final-two-minutes.jsonl').write_text(''.join(json.dumps(d,separators=(',',':'))+'\n' for d in ending))
 events=[]
 for name,contents in text_files.items():
  if not name.startswith('trashcan-platform/events.jsonl'):continue
  for line in contents.splitlines():
   try:d=json.loads(line)
   except ValueError:continue
   if d.get('boot')==boot:events.append(d)
 (root/h/'prior-platform-events.jsonl').write_text(''.join(json.dumps(d,separators=(',',':'))+'\n' for d in events))
 (root/h/'current-platform-events.jsonl').write_text(''.join(line+'\n' for line in text_files['trashcan-platform/events.jsonl'].splitlines() if json.loads(line).get('boot')==(root/h/'state.txt').read_text().splitlines()[2]))
 report['recorders'][h]={'prior_boot':boot,'last_telemetry':latest,'invalid_json_lines':bad,'prior_platform_event_count':len(events),'prior_platform_last_utc':events[-1]['utc'] if events else None,'dump_files':dumps}
 before=(root/h/'interrupted-workspace/slab-governors.before').read_text().splitlines()
 current=[x for x in (root/h/'state.txt').read_text().splitlines() if x.startswith('/sys/devices/system/cpu/')]
 report['governors'][h]={'saved_count':len(before),'current_count':len(current),'identical':before==current,'current_values':sorted(set(x.split()[-1] for x in current))}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'slab':report['slab'],'governors':report['governors'],'recorders':{h:{'last_telemetry_utc':r['last_telemetry']['utc'],'events':r['prior_platform_event_count'],'last_event_utc':r['prior_platform_last_utc'],'dump_files':r['dump_files']} for h,r in report['recorders'].items()}},indent=2))
