#!/usr/bin/env python3
"""Summarize ab.sh runs: per configuration (the run name without its trailing repetition number), the prefill and
decode rates of every repetition, host phases per generated token, MTP round costs, whether the generated tokens equal
those of the default configuration of the same phase, the qwen-check verdicts, and whether every card held 850/1370
MHz while computing.

usage: summarize.py RESULTS_DIR [PREFIX ...]      (RESULTS_DIR: the workspace's results/, pulled from incite)
"""
import collections
import json
import os
import re
import statistics
import sys


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def clocks_ok(err):
    # with-clocks.sh: "[clocks] N samples card0 (..) sclk/mclk: 850/1370 xA 450/1370 xB ... card1 ..."
    if not os.path.exists(err):
        return "?"
    text = open(err).read()
    m = re.findall(r"sclk/mclk: (.*?) max", text)
    if not m:
        return "?"
    full = all("850/1370" in card for card in m)
    return "ok" if full else "LOW"


def fmt(values, digits=1):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.fmean(values):.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root, prefixes = sys.argv[1], sys.argv[2:]
    groups = collections.defaultdict(list)
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or (prefixes and not any(name.startswith(p) for p in prefixes)):
            continue
        run = name[:-5]
        group = re.sub(r"-\d+$", "", run)
        groups[group].append((run, rows(os.path.join(root, name)), clocks_ok(os.path.join(root, run + ".err"))))
    # Tokens of the default configuration per phase and command shape, to compare against.
    reference = {}
    for group, runs in groups.items():
        if "-default" in group or group.startswith("base-"):
            for _, rs, _ in runs:
                for r in rs:
                    if "token_ids" in r:
                        reference.setdefault((group.split("-")[0], r.get("pp"), r.get("tg")), r["token_ids"])
    print(f"{'configuration':28s} {'n':>2s} {'pp tok/s':>22s} {'tg tok/s':>22s} {'exch ms':>7s} {'wait ms':>7s} {'add ms':>6s} "
          f"{'verify ms':>9s} {'tok/rnd':>7s} {'tokens':>7s} {'clocks':>6s}")
    for group in sorted(groups):
        runs = groups[group]
        bench = [r for _, rs, _ in runs for r in rs if "pp_tok_s" in r]
        checks = [r for _, rs, _ in runs for r in rs if "pass" in r and "pp_tok_s" not in r]
        clocks = "ok" if all(c == "ok" for _, _, c in runs) else ",".join(c for _, _, c in runs)
        if bench:
            tp = [r.get("tg_tp_ms") or {} for r in bench]
            mtp = [r.get("mtp") or {} for r in bench]
            same = []
            for r in bench:
                ref = reference.get((group.split("-")[0], r.get("pp"), r.get("tg")))
                same.append("same" if ref == r.get("token_ids") else "DIFF" if ref is not None else "?")
            tokens = "same" if all(s == "same" for s in same) else "/".join(sorted(set(same)))
            print(f"{group:28s} {len(bench):2d} {fmt([r.get('pp_tok_s') for r in bench]):>22s} {fmt([r.get('tg_tok_s') for r in bench], 2):>22s} "
                  f"{fmt([t.get('exchange') for t in tp], 2):>7s} {fmt([t.get('wait') for t in tp], 2):>7s} {fmt([t.get('add') for t in tp], 2):>6s} "
                  f"{fmt([m.get('verify_ms_per_round') for m in mtp], 1):>9s} {fmt([m.get('tokens_per_round') for m in mtp], 2):>7s} {tokens:>7s} {clocks:>6s}")
        for c in checks:
            spec = c.get("speculative") or {}
            print(f"{group:28s} check pass={c.get('pass')} restore_exact={c.get('restore_exact')} "
                  f"column_mismatches={len(c.get('column_mismatches') or [])} max_column_score_error={c.get('max_column_score_error')} "
                  f"speculative_identical={spec.get('identical')} acceptance={(spec.get('stats') or {}).get('acceptance')} clocks={clocks}")


if __name__ == "__main__":
    main()
