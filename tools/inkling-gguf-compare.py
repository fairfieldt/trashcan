#!/usr/bin/env python3
"""Check that an Inkling 2 Mini GGUF in Unsloth's root layout (`inkling2`) and one in upstream's Shard_Rewrite layout
(`inkling2_loop`) hold the same tensors: same type, shape and bytes under the name mapping of docs/inkling.md
("GGUF layouts"). The loop message queries are compared column by column (root `loop_res_query` and
`loop_kv_res_query` [2304, 24] against upstream's per-layer `loop_msg_query` [2304, 4]).

usage: inkling-gguf-compare.py SHARD.gguf ROOT.gguf  (needs llama.cpp's gguf-py on PYTHONPATH and numpy)
"""
import hashlib
import re
import sys

import numpy as np
from gguf import GGUFReader

CTRL = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}


def root_name(name):
    m = re.fullmatch(r"blk\.(\d+)\.loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight", name)
    if m:
        return f"blk.{m[1]}.loop_{CTRL[m[2]]}_{m[3]}.weight"
    return name


def digest(t):
    return hashlib.sha256(np.ascontiguousarray(t.data).view(np.uint8).tobytes()).hexdigest()


def main():
    shard, root = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
    S = {t.name: t for t in shard.tensors}
    R = {t.name: t for t in root.tensors}
    used = set()
    same = differ = 0
    for name, t in S.items():
        m = re.fullmatch(r"blk\.(\d+)\.loop_msg_query\.weight", name)
        if m:
            # [2304, 4] in ggml order: data is [4][2304]; root [2304, 24]: data [24][2304].
            i = int(m[1]) - 6
            msg = np.asarray(t.data, dtype=np.float32).reshape(4, -1)
            res = np.asarray(R["loop_res_query.weight"].data, dtype=np.float32).reshape(24, -1)
            kv = np.asarray(R["loop_kv_res_query.weight"].data, dtype=np.float32).reshape(24, -1)
            want = np.stack([res[2 * i], kv[2 * i], kv[2 * i + 1], res[2 * i + 1]])
            used |= {"loop_res_query.weight", "loop_kv_res_query.weight"}
            if msg.tobytes() == want.tobytes():
                same += 1
            else:
                differ += 1
                print(f"DIFFER {name}: max abs {np.abs(msg - want).max():.3e}")
            continue
        rn = root_name(name)
        if rn not in R:
            differ += 1
            print(f"MISSING in root: {name} ({rn})")
            continue
        used.add(rn)
        u = R[rn]
        if t.tensor_type != u.tensor_type or list(t.shape) != list(u.shape):
            differ += 1
            print(f"DIFFER {name}: {t.tensor_type.name} {list(t.shape)} vs {u.tensor_type.name} {list(u.shape)}")
            continue
        if digest(t) != digest(u):
            differ += 1
            print(f"DIFFER {name}: bytes")
            continue
        same += 1
    extra = sorted(set(R) - used)
    for n in extra:
        print(f"ONLY in root: {n}")
    print(f"{same} tensors identical, {differ} different, {len(extra)} only in root "
          f"({len(S)} shard tensors, {len(R)} root tensors)")
    return 1 if differ or extra else 0


if __name__ == "__main__":
    sys.exit(main())
