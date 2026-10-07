"""Expert popularity from the imatrix: per layer, the share of routed-expert activations covered by the most-used
fraction of experts (a static VRAM cache chosen from these counts). usage: uv run --with numpy python3 THIS IMATRIX"""
import sys

sys.path.insert(0, '/Users/fairfieldt/dev/trashcan/llama.cpp/gguf-py')
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(sys.argv[1])
counts = {}
for t in r.tensors:
    if t.name.endswith('ffn_up_exps.weight.counts'):
        il = int(t.name.split('.')[1])
        counts[il] = np.asarray(t.data, dtype=np.float64).reshape(-1)
assert counts, [t.name for t in r.tensors][:20]
fracs = [0.25, 0.4, 0.5, 0.6, 0.65, 0.75, 0.9]
cover = {f: [] for f in fracs}
for il in sorted(counts):
    c = np.sort(counts[il])[::-1]
    total = c.sum()
    for f in fracs:
        cover[f].append(c[:int(round(f * len(c)))].sum() / total)
print(f'{len(counts)} layers, {len(next(iter(counts.values())))} experts, '
      f'{int(sum(v.sum() for v in counts.values()))} activations counted')
for f in fracs:
    v = np.array(cover[f])
    print(f'top {f:.0%} of experts per layer: mean coverage {v.mean():.1%}, worst layer {v.min():.1%}')
