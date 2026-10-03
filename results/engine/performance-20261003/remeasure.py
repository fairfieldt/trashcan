"""Reproduce baseline/optimized microbenchmarks with valid command-buffer lifetimes."""
from pathlib import Path
import os
import subprocess

out=Path('results/engine/performance-20261003')
shaders=Path('engine/shaders')
original={p.name:p.read_bytes() for p in (out/'baseline-shaders').iterdir()}
optimized={name:(shaders/name).read_bytes() for name in original}
env=os.environ.copy()
env['PATH']=str(Path.home()/'.cargo/bin')+':'+env['PATH']
model=str(Path.home()/'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf')

def measure(bin_name, file_name, args):
    with (out/file_name).open('w') as log, (out/(file_name+'.log')).open('w') as err:
        subprocess.run(['sudo','-n','python3','scripts/latency-run.py',str(out/bin_name),*args],stdout=log,stderr=err,check=True)
try:
    for label,source in [('original',original),('optimized',optimized)]:
        for name,body in source.items(): (shaders/name).write_bytes(body)
        with (out/(label+'-build.log')).open('w') as f:
            subprocess.run(['cargo','build','--release'],env=env,stdout=f,stderr=f,check=True)
        subprocess.run(['cp','target/release/trashcan',str(out/(label+'-bin'))],check=True)
        for batch in (1,8):
            measure(label+'-bin',f'selection-{label}-b{batch}.jsonl',['topk-bench','--batch',str(batch)])
        for tensor in ('ffn_gate','ffn_down','attn_q','attn_output'):
            measure(label+'-bin',f'matvec-{label}-{tensor}.jsonl',['matvec-bench',model,'--iterations','1000','--tensor',f'blk.0.{tensor}.weight'])
        for batch in (2,4,8):
            measure(label+'-bin',f'multivec-{label}-b{batch}.jsonl',['matvec-bench',model,'--iterations','1000','--batch',str(batch)])
        print('PASS:',label,'all warmed microbenchmarks',flush=True)
finally:
    for name,body in optimized.items(): (shaders/name).write_bytes(body)
