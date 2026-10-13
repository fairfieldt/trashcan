#!/usr/bin/env python3
import json, sys
for f in sys.argv[1:]:
    j = json.load(open(f))
    print(f"{f.split('/')[-1]:28s} decode wall {j['decode_wall_ms']:.3f}  gpu sum {sum(sum(s['layer_ms']) for s in j['decode']):.3f}"
          f"  per stage {[round(sum(s['layer_ms']), 3) for s in j['decode']]}  head slices {j.get('head_slice_wall_ms')}")
