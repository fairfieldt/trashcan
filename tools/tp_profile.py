#!/usr/bin/env python3
"""Summarize a TRASHCAN_TP_PROFILE file (JSON lines from tp.rs): per rank, where each decode segment's time goes.

usage: tp_profile.py PROFILE.jsonl [--n 1] [--skip 2] [--dispatch] [--interval 4]

Segments of a forward: 0 = layer 0's mixer, then (ffn of layer k, mixer of layer k + 1)..., last = head.
Times in microseconds, medians over the selected steps (decode steps by default, the first --skip dropped).
"""
import argparse
import collections
import json
import statistics as st


def med(v):
    return statistics_or_nan(v, st.median)


def mean(v):
    return statistics_or_nan(v, st.fmean)


def statistics_or_nan(v, f):
    v = list(v)
    return f(v) if v else float("nan")


def segments(labels, ns):
    """(begin, end) GPU ns of every segment, and the dispatch spans inside each."""
    out, cur, opened = [], None, {}
    for lab, t in zip(labels, ns):
        if lab == "begin":
            cur = {"begin": t, "spans": [], "starts": {}}
        elif lab == "end":
            cur["end"] = t
            out.append(cur)
            cur = None
        elif cur is not None:
            # dispatch labels come in pairs: start (top of pipe) then end (bottom of pipe)
            if lab in cur["starts"]:
                cur["spans"].append((lab, cur["starts"].pop(lab), t))
            else:
                cur["starts"][lab] = t
    return out


def kind(i, nseg, interval):
    if i == nseg - 1:
        return "head"
    if i % 2 == 1:
        return "ffn"
    layer = i // 2
    return "attn" if (layer + 1) % interval == 0 else "delta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--n", type=int, default=1, help="batch width of the forwards to keep")
    ap.add_argument("--skip", type=int, default=2, help="drop this many matching steps per rank first")
    ap.add_argument("--interval", type=int, default=4)
    ap.add_argument("--dispatch", action="store_true", help="per-dispatch table")
    ap.add_argument("--draft", action="store_true", help="keep draft steps (3 segments) instead of forwards")
    a = ap.parse_args()

    labels = {}
    rows = collections.defaultdict(list)
    for line in open(a.profile):
        d = json.loads(line)
        key = (d["rank"], d["graph"])
        if "labels" in d:
            labels[key] = d["labels"]
        if d["n"] != a.n:
            continue
        nseg = len(d["release"])
        if a.draft != (nseg <= 3):
            continue
        rows[d["rank"]].append((d, labels[key]))

    for rank in sorted(rows):
        steps = rows[rank][a.skip:]
        if not steps:
            continue
        per_kind = collections.defaultdict(lambda: collections.defaultdict(list))
        totals = collections.defaultdict(list)
        disp = collections.defaultdict(lambda: collections.defaultdict(list))
        for d, labs in steps:
            segs = segments(labs, d["ns"])
            nseg = len(segs)
            h0 = d["host0"]
            rel, done, exch = d["release"], d["done"], d["exchanged"]
            t_total = done[-1] - rel[0]
            totals["step"].append(t_total / 1e3)
            gpu_busy = sum(s["end"] - s["begin"] for s in segs)
            totals["gpu segments"].append(gpu_busy / 1e3)
            totals["launch"].append(sum(h0 + s["begin"] - r for s, r in zip(segs, rel)) / 1e3)
            totals["notice"].append(sum(dn - (h0 + s["end"]) for s, dn in zip(segs, done)) / 1e3)
            totals["host between"].append(sum(rel[i + 1] - done[i] for i in range(nseg - 1)) / 1e3)
            for i, s in enumerate(segs):
                k = kind(i, nseg, a.interval) if nseg > 3 else f"draft{i}"
                pk = per_kind[k]
                pk["launch"].append((h0 + s["begin"] - rel[i]) / 1e3)
                pk["gpu"].append((s["end"] - s["begin"]) / 1e3)
                pk["notice"].append((done[i] - (h0 + s["end"])) / 1e3)
                if i + 1 < nseg:
                    pk["host"].append((rel[i + 1] - done[i]) / 1e3)
                    if i < len(exch):
                        pk["exchange+add"].append((exch[i] - done[i]) / 1e3)
                prev = s["begin"]
                for lab, t0, t1 in s["spans"]:
                    dd = disp[(k, lab)]
                    dd["gap"].append((t0 - prev) / 1e3)
                    dd["exec"].append((t1 - t0) / 1e3)
                    prev = t1
                if s["spans"]:
                    disp[(k, "(tail)")]["gap"].append((s["end"] - prev) / 1e3)
        print(f"rank {rank}: {len(steps)} steps (n={a.n})")
        for name, v in totals.items():
            print(f"  {name:14s} mean {mean(v):9.1f}  median {med(v):9.1f} us")
        print(f"  {'segment':8s} {'count':>5s} " + " ".join(f"{c:>13s}" for c in ["launch", "gpu", "notice", "host", "exchange+add"]))
        for k, pk in per_kind.items():
            cnt = len(pk["gpu"]) // len(steps)
            print(f"  {k:8s} {cnt:5d} " + " ".join(f"{med(pk[c]):13.1f}" for c in ["launch", "gpu", "notice", "host", "exchange+add"]))
        if a.dispatch and disp:
            print(f"  {'segment':8s} {'dispatch':36s} {'count':>5s} {'gap med':>8s} {'exec med':>9s} {'exec sum/step':>13s}")
            for (k, lab), dd in sorted(disp.items(), key=lambda e: -sum(e[1]["exec"]) if e[1]["exec"] else 0):
                cnt = len(dd["gap"]) // len(steps)
                ex = dd["exec"]
                print(f"  {k:8s} {lab:36s} {cnt:5d} {med(dd['gap']):8.1f} {med(ex) if ex else 0:9.1f} {sum(ex) / len(steps) if ex else 0:13.1f}")
            gaps = sum(sum(dd["gap"]) for dd in disp.values()) / len(steps)
            execs = sum(sum(dd["exec"]) for dd in disp.values()) / len(steps)
            print(f"  per step: dispatch exec {execs / 1e3:.2f} ms, gaps between dispatches {gaps / 1e3:.2f} ms")


if __name__ == "__main__":
    main()
