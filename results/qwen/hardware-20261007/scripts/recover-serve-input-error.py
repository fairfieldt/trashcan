#!/usr/bin/env python3
"""Recover only the known invalid replay-input test after proving serial server idleness."""
import importlib.util,json,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
plan=here.parent/'plans/serve-final'
root='/home/fairfieldt/trashcan-exp/tp4-offline-20261007/results/hardware-20261007/serve-budget-final-plain'
out=plan/'input-error-recovery';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('tp_recovery',here/'run-tp4.py');tp=importlib.util.module_from_spec(spec);spec.loader.exec_module(tp)
events=[]
def run(argv,source=None):
 r=subprocess.run(argv,input=source,text=True,capture_output=True,timeout=40)
 events.append(dict(unix=time.time(),argv=argv,source=source,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 (out/'commands.json').write_text(json.dumps(events,indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr or r.stdout)
 return r.stdout
stop='''import json,os,pathlib,signal,time,urllib.request
root=pathlib.Path("ROOT")
scope=json.loads((root/'scope.json').read_text())
client=json.loads((root/'client/client-summary.json').read_text())
assert client['server_identity_verified'] and client['completed_requests']==1
assert 'replayed input items must have completed status' in client['error']
log=(root/'stderr.log').read_text();assert 'request failed:' not in log and 'cleanup failed' not in log
with urllib.request.urlopen('http://127.0.0.1:8017/v1/models',timeout=10) as r:models=json.load(r)
assert any(v['id']=='budget-serve-budget-final-plain' for v in models['data'])
with urllib.request.urlopen('http://127.0.0.1:8017/health',timeout=10) as r:
 assert r.status==200
 health=r.read().decode()
proof=dict(unix=time.time(),reason='One generation completed; next request rejected before generation for invalid replay status; fresh serial health barrier confirms driver checkpoint finished',health=health,models=models,pid=scope['pid'])
p=pathlib.Path('/proc')/str(scope['pid'])
assert p.joinpath('stat').read_text().split()[21]==scope['process_start_ticks']
assert p.joinpath('cmdline').read_bytes().rstrip(b'\\0').split(b'\\0')==[v.encode() for v in scope['command']]
(root/'recovery-idle-stop.json').write_text(json.dumps(proof,indent=2)+'\\n')
os.kill(scope['pid'],signal.SIGTERM)
print(json.dumps(proof))
'''.replace('ROOT',root)
run(['ssh','incite','python3 -'],stop)
for attempt in range(20):
 log=run(['ssh','invite',f'tail -8 {root}/stderr.log'])
 if log.rfind('tp-node: session ended')>log.rfind('tp-node: session over '):break
 time.sleep(1)
else:raise RuntimeError('peer teardown not confirmed; leave leases intact')
run(['ssh','invite',f'python3 - {root}'],tp.STOP_NODE)
for host in ['incite','invite']:
 for attempt in range(20):
  scope=json.loads(run(['ssh',host,f'cat {root}/scope.json']))
  if scope.get('finished_unix'):break
  time.sleep(1)
 else:raise RuntimeError(host+' scope not finished')
 assert scope['returncode'] in (0,-15,143)
 assert scope.get('governors_before') and scope['governors_before']==scope['governors_after'] and not scope.get('restore_errors')
 (out/host).mkdir()
 run(['rsync','-a',f'{host}:{root}/',str(out/host)+'/'])
for host,ticket in [('incite',116),('invite',104)]:
 run(['gpuq','--on',host,'release',str(ticket),'-r','Invalid replay-input harness recovered after fresh idle barrier; both scopes finished and governors restored'])
summary=json.loads((plan/'summary.json').read_text())
summary.update(state='stopped; invalid replay-input harness; recovered and released',active_tickets={},recovery=dict(path=str(out),finished_unix=time.time(),reason='HTTP400 from incomplete replay status; no model correctness result'))
(plan/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Recovered idle processes; governors restored; tickets116/104 released.')
