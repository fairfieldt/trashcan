#!/usr/bin/env python3
"""Summarize bench.sh output directories: python3 summarize.py DIR..."""
import glob, json, os, re, statistics, sys


def gbps(f, key="sum_received"):
    try:
        j = json.load(open(f))
    except Exception:
        return None
    e = j.get("end", {})
    if key not in e:
        return None
    return e[key]["bits_per_second"] / 1e9


def retr(f):
    try:
        return json.load(open(f))["end"]["sum_sent"].get("retransmits")
    except Exception:
        return None


def sockperf(f):
    s = open(f).read() if os.path.exists(f) else ""
    def pct(p):
        m = re.search(r"percentile %s\s*=\s*([\d.]+)" % re.escape(p), s)
        return float(m.group(1)) if m else None
    return pct("50.000"), pct("99.000")


def ping(f):
    s = open(f).read() if os.path.exists(f) else ""
    m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", s)
    return (float(m.group(2)) * 1000, float(m.group(1)) * 1000) if m else (None, None)


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{statistics.median(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})"


print("| config | sockperf p50 / p99 us | ping avg / min us | fwd Gb/s | rev Gb/s | bidir Gb/s | 4 streams Gb/s | fwd retr |")
print("|---|---|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    p50, p99 = sockperf(f"{d}/sockperf-64B.txt")
    pa, pm = ping(f"{d}/ping-56B.txt")
    fwd = [gbps(f) for f in sorted(glob.glob(f"{d}/tcp-fwd-*.json"))]
    rev = [gbps(f) for f in sorted(glob.glob(f"{d}/tcp-rev-*.json"))]
    bi = f"{d}/tcp-bidir.json"
    b1, b2 = gbps(bi), gbps(bi, "sum_received_bidir_reverse")
    bis = f"{b1:.2f} + {b2:.2f}" if b1 and b2 else "n/a"
    p4 = gbps(f"{d}/tcp-p4.json")
    rt = [retr(f) for f in sorted(glob.glob(f"{d}/tcp-fwd-*.json"))]
    s = lambda v: f"{v:.1f}" if v is not None else "n/a"
    print(f"| {os.path.basename(d.rstrip('/'))} | {s(p50)} / {s(p99)} | {s(pa)} / {s(pm)} | {fmt(fwd)} | {fmt(rev)} | {bis} | {fmt([p4])} | {','.join(str(x) for x in rt)} |")
