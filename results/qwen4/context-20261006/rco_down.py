"""Which layers ISTA's RCO allocations put at Q2_0 for ffn_down_exps, per build (least sensitive first)."""
import csv

path = '/Users/fairfieldt/dev/claude-work/trashcan-20261006/quant-survey/ista-expert-byte-ranges.tsv'
builds = {}
with open(path) as f:
    for row in csv.reader((l for l in f if not l.startswith('#')), delimiter='\t'):
        variant, shard, tensor, ty = row[0], row[1], row[2], row[3]
        if 'ffn_down_exps' not in tensor:
            continue
        layer = int(tensor.split('.')[1])
        builds.setdefault(variant, {})[layer] = ty
for v in ('IQ3_S', 'IQ3_XXS', 'IQ2_XS', 'Q2_0'):
    q2 = [l for l, t in sorted(builds[v].items()) if t == 'Q2_0']
    print(f'{v}: {len(q2)} layers at Q2_0 down: {q2}')
# One layer per GPU (groups of 4): prefer layers at Q2_0 in the IQ3_S build, then the IQ3_XXS build.
pick = []
for g in range(12):
    group = range(4 * g, 4 * g + 4)
    s = [l for l in group if builds['IQ3_S'].get(l) == 'Q2_0']
    x = [l for l in group if builds['IQ3_XXS'].get(l) == 'Q2_0']
    pick.append((s or x or [None])[0])
print('one per GPU:', pick)
