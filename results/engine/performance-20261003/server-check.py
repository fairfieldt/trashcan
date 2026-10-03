"""Run baseline/final resident servers sequentially and compare response latency and text."""
import argparse
import contextlib
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',required=True)
parser.add_argument('--optimized',required=True)
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
out=Path('results/engine/performance-20261003')
model=str(Path.home()/'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf')
prompts=[('chat','Explain how a transformer works, step by step, in English.'),
('chat-photosynthesis','Explain photosynthesis to a ten-year-old in three short paragraphs.'),
('chat-csv','Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.')]
texts={}
for index, (label, binary, draft) in enumerate([('baseline',args.baseline,False),('optimized',args.optimized,False),('baseline-mtp',args.baseline,True),('optimized-mtp',args.optimized,True)]):
    port=18004+index
    if args.resume and (out/f'server-{label}-timing.json').exists():
        if label=='baseline':
            for name,_ in prompts:
                payload=json.loads((out/f'server-{label}-{name}.json').read_text())
                texts[name]=''.join(part['text'] for item in payload['output'] if item['type']=='message' for part in item['content'])
        print('Retained completed run:',label,flush=True)
        continue
    logfile=out/f'server-{label}.log'
    with logfile.open('w') as log:
        command=['sudo','-n','python3','scripts/latency-run.py',binary,'serve',model,'--port',str(port),'--batch','128']
        if draft: command += ['--draft',str(Path.home()/'models/gemma-4-12b-assistant/model.safetensors')]
        p=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
        try:
            deadline=time.monotonic()+180
            while 'Responses server listening' not in logfile.read_text():
                if p.poll() is not None: raise RuntimeError('server exited: '+logfile.read_text())
                if time.monotonic()>deadline: raise RuntimeError('server startup timeout')
                time.sleep(0.5)
            rows=[]
            for name,prompt in prompts:
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
                body=json.dumps({'model':'gemma-4-12b','input':prompt,'temperature':0,'max_output_tokens':128,'store':False})
                start=time.monotonic()
                c.request('POST','/v1/responses',body,{'Content-Type':'application/json'})
                response=c.getresponse()
                payload=json.loads(response.read())
                seconds=time.monotonic()-start
                c.close()
                assert response.status==200,payload
                text=''.join(part['text'] for item in payload['output'] if item['type']=='message' for part in item['content'])
                if label=='baseline': texts[name]=text
                else: assert text==texts[name],name
                (out/f'server-{label}-{name}.json').write_text(json.dumps(payload,indent=2)+'\n')
                rows.append({'prompt':name,'seconds':seconds,'output_tokens':payload['usage']['output_tokens'],'text_matches_baseline':label!='baseline'})
                print(label,name,round(seconds,3),'seconds',flush=True)
            (out/f'server-{label}-timing.json').write_text(json.dumps(rows,indent=2)+'\n')
            if label.startswith('optimized'):
                with (out/f'server-validation-{label}.log').open('w') as check:
                    subprocess.run(['python3','scripts/test-server.py','--url',f'http://127.0.0.1:{port}'],stdout=check,stderr=check,check=True)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid,signal.SIGTERM)
            p.wait(timeout=30)
print('PASS: baseline/final text parity and Responses integration suite',flush=True)
