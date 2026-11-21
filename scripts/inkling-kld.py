#!/usr/bin/env python3
"""KL divergence between two sets of full-vocabulary logits over the same token chunks.

Each set is a directory of chunk-NNN.f32 files (CTX rows of n_vocab f32 logits, row i predicting token i + 1), as
written by tools/dump-logits (llama.cpp) and `trashcan inkling logits` (the engine). The reference P may instead be a
llama-perplexity `--kl-divergence-base` file (`--base FILE`), which holds log-probabilities quantized to 16 bits for
the second half of every chunk (positions CTX/2 .. CTX-2), as llama.cpp's own KLD uses them.

For every position: KLD(P || Q) = sum_i p_i (log p_i - log q_i) over the whole vocabulary in float64, whether the two
argmaxes agree, the difference of the next token's probability, and the logit difference d = q - p summarised by
its P-weighted mean (a uniform shift, which the softmax removes) and its P-weighted standard deviation (what remains).

usage: inkling-kld.py P_DIR|--base FILE Q_DIR --ids IDS [--chunks 0,1,..] [--out PREFIX] [--jobs N]
Writes PREFIX.json (summary) and PREFIX.npz (per-position values) when --out is given.
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
from multiprocessing import Pool

import numpy as np

BLOCK = 128


def meta(d):
    m = {}
    for line in open(os.path.join(d, "meta.tsv")):
        k, _, v = line.rstrip("\n").partition("\t")
        m[k] = v
    return m


def log_softmax(x):
    x = x.astype(np.float64)
    mx = x.max(axis=1, keepdims=True)
    return x - (mx + np.log(np.exp(x - mx).sum(axis=1, keepdims=True)))


class Base:
    """A llama-perplexity logits file: header, tokens, then per chunk CTX - 1 - CTX/2 rows of
    [scale f32, min_log_prob f32, n_vocab uint16 (padded to even)]."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            assert f.read(8) == b"_logits_"
            self.ctx, self.vocab, self.chunks = struct.unpack("<iii", f.read(12))
        self.first = self.ctx // 2
        self.rows = self.ctx - 1 - self.first
        self.nv = 2 * ((self.vocab + 1) // 2) + 4
        self.offset = 20 + 4 * self.ctx * self.chunks

    def logp(self, chunk):
        a = np.memmap(self.path, dtype=np.uint16, mode="r", offset=self.offset + chunk * self.rows * self.nv * 2,
                      shape=(self.rows, self.nv))
        head = np.ascontiguousarray(a[:, :4]).view(np.float32)
        scale, lo = head[:, 0:1].astype(np.float64), head[:, 1:2].astype(np.float64)
        return scale * a[:, 4:4 + self.vocab].astype(np.float64) + lo


def chunk_stats(args):
    p_src, q_dir, c, ids, ctx, vocab = args
    q = np.memmap(os.path.join(q_dir, f"chunk-{c:03d}.f32"), dtype=np.float32, mode="r").reshape(ctx, vocab)
    if isinstance(p_src, str):
        p = np.memmap(os.path.join(p_src, f"chunk-{c:03d}.f32"), dtype=np.float32, mode="r").reshape(ctx, vocab)
        rows = range(0, ctx - 1)
        get_p = lambda a, b: log_softmax(p[a:b])
        p_raw = lambda a, b: p[a:b].astype(np.float64)
    else:
        base = Base(p_src)
        lp_all = base.logp(c)
        rows = range(base.first, ctx - 1)
        get_p = lambda a, b: lp_all[a - base.first:b - base.first]
        p_raw = get_p  # log-probabilities: the logit differences are then shifted by a constant per row
    out = {k: [] for k in ("pos", "kld", "same", "dp", "shift", "spread", "dmax", "ptop", "lse_p", "lse_q")}
    r0, r1 = rows.start, rows.stop
    for a in range(r0, r1, BLOCK):
        b = min(a + BLOCK, r1)
        lp = get_p(a, b)
        qq = q[a:b].astype(np.float64)
        mq = qq.max(axis=1, keepdims=True)
        lse_q = mq + np.log(np.exp(qq - mq).sum(axis=1, keepdims=True))
        lq = qq - lse_q
        pp = np.exp(lp)
        kld = (pp * (lp - lq)).sum(axis=1)
        nxt = ids[c * ctx + a + 1:c * ctx + b + 1]
        idx = np.arange(b - a)
        dp = np.exp(lq[idx, nxt]) - pp[idx, nxt]
        raw = p_raw(a, b)
        d = qq - raw
        shift = (pp * d).sum(axis=1)
        spread = np.sqrt(np.maximum((pp * (d - shift[:, None]) ** 2).sum(axis=1), 0))
        mp = raw.max(axis=1, keepdims=True)
        lse_p = (mp + np.log(np.exp(raw - mp).sum(axis=1, keepdims=True)))[:, 0]
        out["pos"].append(np.arange(a, b))
        out["kld"].append(kld)
        out["same"].append(lp.argmax(axis=1) == lq.argmax(axis=1))
        out["dp"].append(dp)
        out["shift"].append(shift)
        out["spread"].append(spread)
        out["dmax"].append(qq.max(axis=1) - raw.max(axis=1))
        out["ptop"].append(pp.max(axis=1))
        out["lse_p"].append(lse_p)
        out["lse_q"].append(lse_q[:, 0])
    r = {k: np.concatenate(v) for k, v in out.items()}
    r["chunk"] = np.full(len(r["pos"]), c)
    return r


def pct(x, q):
    return float(np.percentile(x, q))


def summary(r, sel):
    k = r["kld"][sel]
    return {
        "positions": int(sel.sum()),
        "mean_kld": float(k.mean()),
        "mean_kld_stderr": float(k.std(ddof=1) / np.sqrt(len(k))),
        "median_kld": pct(k, 50),
        "p90_kld": pct(k, 90),
        "p99_kld": pct(k, 99),
        "p99.9_kld": pct(k, 99.9),
        "max_kld": float(k.max()),
        "same_top": float(r["same"][sel].mean()),
        "rms_dp": float(np.sqrt((r["dp"][sel] ** 2).mean())),
        "mean_shift": float(r["shift"][sel].mean()),
        "mean_abs_shift": float(np.abs(r["shift"][sel]).mean()),
        "mean_spread": float(r["spread"][sel].mean()),
        "mean_dmax": float(r["dmax"][sel].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p", nargs="?")
    ap.add_argument("q")
    ap.add_argument("--base")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--chunks")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.base and a.p:
        a.q, a.p = a.p, None
    mq = meta(a.q)
    ctx, vocab = int(mq["ctx"]), int(mq["vocab"])
    if a.p:
        mp = meta(a.p)
        assert (int(mp["ctx"]), int(mp["vocab"])) == (ctx, vocab), "the two sets differ in chunk size or vocabulary"
    else:
        b = Base(a.base)
        assert (b.ctx, b.vocab) == (ctx, vocab), "the base file differs in chunk size or vocabulary"
    ids = np.array(open(a.ids).read().split(), dtype=np.int64)
    have = sorted(int(re.search(r"chunk-(\d+)", f)[1]) for f in glob.glob(os.path.join(a.q, "chunk-*.f32")))
    chunks = [int(c) for c in a.chunks.split(",")] if a.chunks else have
    src = a.p if a.p else a.base
    with Pool(a.jobs) as pool:
        parts = pool.map(chunk_stats, [(src, a.q, c, ids, ctx, vocab) for c in chunks])
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    every = np.ones(len(r["kld"]), bool)
    second = r["pos"] >= ctx // 2
    res = {
        "p": src, "q": a.q, "chunks": chunks, "ctx": ctx,
        "all": summary(r, every),
        "second_half": summary(r, second),
        "by_position": [dict(start=s, **summary(r, (r["pos"] >= s) & (r["pos"] < s + 256))) for s in range(0, ctx, 256)
                        if ((r["pos"] >= s) & (r["pos"] < s + 256)).any()],
        "by_chunk": [dict(chunk=c, **summary(r, r["chunk"] == c)) for c in chunks],
    }
    print(json.dumps({k: res[k] for k in ("p", "q", "all", "second_half")}, indent=1))
    print("by position (start, mean KLD, same top):")
    for b in res["by_position"]:
        print(f"  {b['start']:5d} {b['mean_kld']:.3e} {b['same_top']:.4f}")
    print("by chunk (chunk, mean KLD, same top):")
    for b in res["by_chunk"]:
        print(f"  {b['chunk']:3d} {b['mean_kld']:.3e} {b['same_top']:.4f}")
    if a.out:
        json.dump(res, open(a.out + ".json", "w"), indent=1)
        np.savez_compressed(a.out + ".npz", **r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
