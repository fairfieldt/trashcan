#!/usr/bin/env python3
"""Mean ms per dispatch label (summed over the 8 layers of both stages, and per occurrence) of stage-bench prefill
profiles; deferred groups are listed as one span under the first member's label + '+'.
usage: disp.py RUN.json [RUN.json ...]"""
import json, sys, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from prof import timeline, groups

for path in sys.argv[1:]:
    j = json.load(open(path))
    tot = collections.OrderedDict()
    cnt = collections.Counter()
    for stage in j['prefill_dispatches']:
        for layer, labs, s, e in groups(timeline(stage)):
            if layer is None:
                continue
            lab = ' + '.join(labs)
            tot[lab] = tot.get(lab, 0) + (e - s)
            cnt[lab] += 1
    print(f"== {path.split('/')[-1]} batch {j['batch']}")
    for lab, ms in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"   {lab:75s} {ms:8.3f} ms  x{cnt[lab]:2d}  {ms / cnt[lab] * 1000:8.1f} us each")
