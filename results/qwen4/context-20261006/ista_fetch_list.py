"""Byte ranges of the ISTA GSQ-RCO Q2_0 down projections to splice, one layer per GPU of six Macs: layers at Q2_0
in the IQ3_S build (RCO's least sensitive), else in the IQ3_XXS build. Writes a TSV for ista_fetch.sh."""
import csv

src = '/Users/fairfieldt/dev/claude-work/trashcan-20261006/quant-survey/ista-expert-byte-ranges.tsv'
out = '/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ista-fetch.tsv'
rows = {}
with open(src) as f:
    for r in csv.reader((l for l in f if not l.startswith('#')), delimiter='\t'):
        variant, shard, tensor, ty, dims, start, end, size = r
        if 'ffn_down_exps' in tensor:
            rows[(variant, int(tensor.split('.')[1]))] = (shard, tensor, ty, int(start), int(end), int(size))
picks = []
for g in range(12):
    for variant in ('IQ3_S', 'IQ3_XXS'):
        ls = [l for l in range(4 * g, 4 * g + 4) if rows[(variant, l)][2] == 'Q2_0']
        if ls:
            picks.append((variant, ls[0]))
            break
with open(out, 'w') as f:
    for variant, layer in picks:
        shard, tensor, ty, start, end, size = rows[(variant, layer)]
        assert ty == 'Q2_0' and size == end - start + 1 == 640 * 2560 * 512 // 64 * 18
        f.write(f'{layer}\t{variant}\t{shard}\t{tensor}\t{start}\t{end}\t{size}\n')
        print(layer, variant, shard, start, size)
