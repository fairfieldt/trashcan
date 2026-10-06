#!/usr/bin/env python3
"""Fetch only the MTP head (plus embed/lm_head/final norm) of a sharded HF safetensors repo via HTTP ranges.

usage: tools/fetch-mtp.py REPO OUTDIR   (e.g. Qwen/Qwen3.5-9B; then convert_hf_to_gguf.py --mtp OUTDIR)
Writes OUTDIR/model.safetensors (selected tensors), model.safetensors.index.json, config and tokenizer files.
"""
import json, struct, sys, urllib.request, os

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
base = f"https://huggingface.co/{repo}/resolve/main/"


def get(path, rng=None):
    req = urllib.request.Request(base + path)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]-1}")
    with urllib.request.urlopen(req) as r:
        return r.read()


for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
          "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json"]:
    try:
        data = get(f)
    except Exception as e:
        print("skip", f, e)
        continue
    open(os.path.join(out, f), "wb").write(data)
    print("fetched", f)

index = json.loads(get("model.safetensors.index.json"))
wm = index["weight_map"]
want = [k for k in wm if k.startswith("mtp.") or k.startswith("model.mtp.")]
want += [k for k in wm if k in ("lm_head.weight", "model.language_model.embed_tokens.weight",
                                 "model.language_model.norm.weight", "model.embed_tokens.weight",
                                 "model.norm.weight")]
headers = {}
for shard in sorted({wm[k] for k in want}):
    n = struct.unpack("<Q", get(shard, (0, 8)))[0]
    headers[shard] = (8 + n, json.loads(get(shard, (8, 8 + n))))

meta, blobs, offset = {}, [], 0
for k in sorted(want):
    shard = wm[k]
    start, h = headers[shard]
    t = h[k]
    a, b = t["data_offsets"]
    print(f"{k}: {t['dtype']} {t['shape']} {(b - a) / 2**20:.1f} MiB from {shard}", flush=True)
    data = get(shard, (start + a, start + b))
    assert len(data) == b - a
    meta[k] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [offset, offset + len(data)]}
    blobs.append(data)
    offset += len(data)

hdr = json.dumps(meta, separators=(",", ":")).encode()
hdr += b" " * (-len(hdr) % 8)
with open(os.path.join(out, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hdr)))
    f.write(hdr)
    for blob in blobs:
        f.write(blob)
json.dump({"metadata": {"total_size": offset}, "weight_map": {k: "model.safetensors" for k in meta}},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print("done", offset / 2**30, "GiB")
