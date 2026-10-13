#!/usr/bin/env python3
"""The out-projection dispatches (2560 rows x 6144) and their part sums in stage-bench's `prefill_dispatches`
(TRASHCAN_Q4_PROFILE=1), per JSON line. usage: outproj.py STAGE-BENCH.json..."""
import json, sys

for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith('{'):
            continue
        d = json.loads(line)
        rows = [x for st in d['prefill_dispatches'] for x in st]
        outp = [x for x in rows if x[0].endswith('2560x6144')]
        sums = [x for x in rows if x[0].startswith('qwen4_sum_parts')]
        print(path, 'batch', d['batch'], 'prefill GPU ms', round(sum(sum(s['layer_ms']) for s in d['prefill']), 2),
              '| out projections', len(outp), sorted({x[0] for x in outp}), round(sum(x[1] for x in outp), 3), 'ms',
              '| part sums', len(sums), round(sum(x[1] for x in sums), 3), 'ms')
