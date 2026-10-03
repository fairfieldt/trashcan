"""Summarize the retained performance measurements; run after all checks finish."""
from pathlib import Path
import json
import re
import statistics

root=Path(__file__).resolve().parent

def read(name): return json.loads((root/name).read_text())
def lines(name): return [json.loads(l) for l in (root/name).read_text().splitlines()]
def pair(a,b): return {'baseline':a,'optimized':b,'speedup':a/b}
summary={'date':'2026-10-03','baseline_commit':'f63fe89','matvec':{},'selection':{},'multivec':{},'chat':{}}
for tensor in ('ffn_gate','ffn_down','attn_q','attn_output'):
    values=[statistics.mean(d['original_GB_s'] for d in lines(f'matvec-{v}-{tensor}.jsonl')) for v in ('original','optimized')]
    summary['matvec'][tensor]={'baseline_GB_s':values[0],'optimized_GB_s':values[1],'speedup':values[1]/values[0]}
for batch in (1,8):
    values=[statistics.mean(d['milliseconds'] for d in lines(f'selection-{v}-b{batch}.jsonl') if d['flags']==3 and d['top_k']==64) for v in ('original','optimized')]
    summary['selection'][f'batch_{batch}_top64_ms']=pair(*values)
for batch in (2,4,8):
    values=[statistics.mean(d['milliseconds'] for d in lines(f'multivec-{v}-b{batch}.jsonl')) for v in ('original','optimized')]
    summary['multivec'][f'batch_{batch}_ms']=pair(*values)
a,b=read('baseline/bench.json'),read('final/bench.json')
summary['bench']={'baseline':a,'optimized':b,'decode_speedup':b['tg_tok_s']/a['tg_tok_s'],'prefill_speedup':b['pp_tok_s']/a['pp_tok_s']}
for prompt in ('chat','chat-photosynthesis','chat-csv'):
    row={}
    for mode in ('target','mtp'):
        values=[]
        for variant in ('baseline','final'):
            log=(root/variant/f'{prompt}-{mode}.log').read_text()
            m=re.search(r'decode: (\d+) tokens, ([\d.]+) tok/s',log)
            assert m and int(m[1])==128
            values.append(float(m[2]))
        assert (root/'baseline'/f'{prompt}-target.txt').read_bytes()==(root/'final'/f'{prompt}-target.txt').read_bytes()
        assert (root/'final'/f'{prompt}-target.txt').read_bytes()==(root/'final'/f'{prompt}-mtp.txt').read_bytes()
        row[mode]={'baseline_tok_s':values[0],'optimized_tok_s':values[1],'speedup':values[1]/values[0],'text_identical':True}
    summary['chat'][prompt]=row
summary['chat']['combined_mtp_gain']={v:sum(128/d['target'][v+'_tok_s'] for d in summary['chat'].values())/sum(128/d['mtp'][v+'_tok_s'] for d in summary['chat'].values()) for v in ('baseline','optimized')}
summary['selector']={'checks':read('selector-check.json'),'batches':{}}
selector_data=lines('selector-sweep.jsonl')
for batch in (5,6,7,8):
    values=[statistics.mean(d['milliseconds'] for d in selector_data if d['batch']==batch and d['kernel']==k) for k in ('multivec','multivec4')]
    summary['selector']['batches'][str(batch)]=pair(*values)
assert (root/'selector-before-chat.txt').read_bytes()==(root/'selector-after-chat.txt').read_bytes()
values=[]
for variant in ('before','after'):
    m=re.search(r'decode: (\d+) tokens, ([\d.]+) tok/s',(root/f'selector-{variant}-chat.log').read_text())
    assert m and int(m[1])==128
    values.append(float(m[2]))
summary['selector']['seven_draft_chat']={'before_tok_s':values[0],'after_tok_s':values[1],'speedup':values[1]/values[0],'text_identical':True}
summary['long_check']=read('long-check.json')
validation=(root/'validation.log').read_text()
summary['validation']={'reference_paths_passed':validation.count('PASS: 132 required tensors, logits, and 10 greedy tokens'),
'worst_relative_activation_error':max(float(v) for v in re.findall(r'worst relative error over 132 tensors: ([\d.e+-]+)',validation))}
assert summary['validation']['reference_paths_passed']==6
summary['server']={}
for mode in ('','-mtp'):
    data={'baseline':read(f'server-baseline{mode}-timing.json'),'optimized':read(f'server-optimized{mode}-timing.json')}
    for a,b in zip(data['baseline'],data['optimized']):
        b['speedup']=a['seconds']/b['seconds']
        b['latency_reduction']=1-b['seconds']/a['seconds']
    summary['server']['mtp' if mode else 'ordinary']=data
summary['validation']['server_ordinary_pass']=(root/'server-validation-optimized.log').read_text().rstrip().endswith('Responses server checks passed')
summary['validation']['server_mtp_pass']=(root/'server-validation-optimized-mtp.log').read_text().rstrip().endswith('Responses server checks passed')
assert summary['validation']['server_ordinary_pass'] and summary['validation']['server_mtp_pass']
summary['repeat']={'baseline':read('baseline-repeat.json'),'optimized':read('optimized-repeat.json')}
summary['repeat']['decode_speedup']=summary['repeat']['optimized']['tg_tok_s']/summary['repeat']['baseline']['tg_tok_s']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
