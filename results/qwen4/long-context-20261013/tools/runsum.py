#!/usr/bin/env python3
"""One line per chain bench run: prefill and decode speed, the driver's last-chunk GPU ms per stage.
usage: runsum.py RUN_DIR... (each holding NAME.json from fetch.sh)"""
import json, os, sys
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip('/'))
    p = os.path.join(d, name + '.json')
    try:
        r = json.load(open(p))
    except Exception as e:
        print(f"{name:28s} no result ({e.__class__.__name__})")
        continue
    last = [round(sum(x), 1) for x in r.get('prefill_last_chunk_layer_ms', [])]
    dec = [round(sum(x), 2) for x in r.get('decode_layer_ms', [])]
    print(f"{name:28s} pp {r['pp']:5d} {r['pp_tok_s']:7.1f} tok/s ({r['pp_seconds']:6.2f} s)  tg {r['tg_tok_s']:6.2f} tok/s"
          f"  last chunk GPU ms {last}  decode GPU ms {dec}")
