#!/usr/bin/env python3
"""One line per chain bench JSON: prefill and decode rates, the driver's GPU ms for its layers per decode token."""
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    print(f"{f.rsplit('/', 1)[-1][:-5]:28s} pp{d['pp']} {d['pp_tok_s']:6.1f} tok/s  tg{d['tg']} {d['tg_tok_s']:5.2f} tok/s "
          f"({1000 / d['tg_tok_s']:5.1f} ms/token)  driver layers {sum(map(sum, d['decode_layer_ms'])):5.2f} ms")
