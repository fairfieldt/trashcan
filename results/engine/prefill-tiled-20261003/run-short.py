#!/usr/bin/env python3
from pathlib import Path
import subprocess
r=Path('results/engine/prefill-tiled-20261003')
m=Path.home()/'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
def run(label,args,binary='candidate',env=None):
 command=['sudo','-n','python3','scripts/latency-run.py']
 if env: command+=['env',*env]
 command += [str(r/'bin'/binary),*map(str,args)]
 with (r/f'{label}.out').open('w') as out,(r/f'{label}.log').open('w') as err:
  subprocess.run(command,stdout=out,stderr=err,check=True)
 print('PASS',label,flush=True)
for position,batch,sliding in [(0,64,False),(37,127,False),(4096,512,False),(80000,512,False),(0,64,True),(1000,127,True),(1531,512,True),(80000,512,True)]:
 run(f'micro-p{position}-b{batch}-s{int(sliding)}',['prefill-attention-bench','--position',position,'--batch',batch,'--iterations',2]+(['--sliding'] if sliding else []))
for rep in [1,2]:
 for binary in (['baseline','candidate'] if rep==1 else ['candidate','baseline']):
  for pp in [512,4096]:
   label=f'{binary}-pp{pp}-{rep}'
   run(label,['bench',m,'--pp',pp,'--tg',128,'--batch',512,'--progress','--profile',r/f'{label}.tsv'],binary)
for batch in [64,128,512]:
 run(f'check-b{batch}',['check-batches',m,'--tokens',1700,'--batch',batch])
print('PASS short suite',flush=True)
