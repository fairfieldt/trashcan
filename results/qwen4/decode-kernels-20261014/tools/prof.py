#!/usr/bin/env python3
"""Per-dispatch decode profile of a stage-bench JSON (TRASHCAN_Q4_PROFILE=1): per layer, every dispatch's mean µs,
the gap before it, its bytes (weights and state it streams), GB/s, and its code size (from a RADV stats dump).
usage: prof.py RUN.json [--layers 5,7] [--stats STATS.txt --vulkan VULKAN_RS] [--compare OTHER.json]"""
import json, re, sys, argparse

G = dict(d=2560, w=10240, rank=320, ff=640, slots=11)


def q8(rows, k):
    return rows * k // 32 * 34


def bytes_of(label):
    """Bytes a dispatch reads from DRAM (weights, recurrent state), by its label."""
    m = re.match(r'(\S+) (\d+)x(\d+)', label)
    if not m:
        return 0
    name, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    if name.startswith('qwen4_mix_down_skw') or name.startswith('qwen4_mix_down_sk'):
        return q8(G['rank'] + 4, G['w'])
    if name.startswith('qwen4_mix_rows_up') or name.startswith('qwen4_mix_up'):
        return q8(G['w'], G['rank'])
    if name.startswith('qwen4_q8_'):
        return q8(a, b)
    if name.startswith('qwen4_matf32') or name.startswith('qwen4_router'):
        return a * b * 4
    if name.startswith('qwen4_moe_gu'):
        return G['slots'] * 2 * G['ff'] * G['d'] // 256 * 136
    if name.startswith('qwen4_moe_down_w'):
        return G['slots'] * G['d'] * G['ff'] // 32 * 18
    if name == 'qwen4_gdn':
        return 48 * 128 * 128 * 4 * 2
    return 0


def code_sizes(stats, vulkan):
    if not stats:
        return {}
    src = open(vulkan).read()
    start = src.index('for (name, bytes) in [')
    end = src.index('] {', start)
    names = re.findall(r'\(\s*"([A-Za-z0-9_]+)",\s*include_bytes!', src[start:end])
    vstart = src.index("let variants: &[(&'static str, Option<u32>)] = match name {")
    vend = src.index('_ => &[(name, None)],', vstart)
    variants = {}
    for m in re.finditer(r'"([A-Za-z0-9_]+)" => &\[(.*?)\],', src[vstart:vend], re.S):
        variants[m.group(1)] = re.findall(r'\("([A-Za-z0-9_]+)",', m.group(2))
    order = []
    for n in names:
        order += variants.get(n, [n])
    blocks = open(stats).read().split('*** SHADER STATS ***')[1:]
    out = {}
    for name, b in zip(order, blocks):
        m = re.search(r'^Code size: (\d+)', b, re.M)
        out[name] = int(m.group(1)) if m else 0
    return out


def layers(run):
    """Per stage: list of (layer, [(label, us, gap_us)])."""
    out = []
    for stage in run['dispatches']:
        cur, ls = None, []
        for label, ms, gap in stage:
            if label.startswith('layer '):
                cur = (int(label.split()[1]), [])
                ls.append(cur)
                cur[1].append(('(layer start)', 0.0, gap * 1e3))
                continue
            if cur is not None:
                cur[1].append((label, ms * 1e3, gap * 1e3))
        out.append(ls)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--layers', default='')
    ap.add_argument('--stats')
    ap.add_argument('--vulkan')
    ap.add_argument('--compare')
    a = ap.parse_args()
    run = json.loads(open(a.run).read().strip().split('\n')[-1])
    other = json.loads(open(a.compare).read().strip().split('\n')[-1]) if a.compare else None
    code = code_sizes(a.stats, a.vulkan)
    want = [int(x) for x in a.layers.split(',') if x]
    dec = run['decode']
    print(f"{a.run}: decode wall {run['decode_wall_ms']:.3f} ms, layer ms {[round(x, 3) for x in dec[0]['layer_ms']]}")
    if other:
        print(f"{a.compare}: decode wall {other['decode_wall_ms']:.3f} ms, layer ms "
              f"{[round(x, 3) for x in other['decode'][0]['layer_ms']]}")
    ol = layers(other) if other else None
    for si, stage in enumerate(layers(run)):
        for li, (layer, ds) in enumerate(stage):
            if want and layer not in want:
                continue
            # Deferred groups (no barrier between members) overlap: their gaps wrap around; a group's time is about
            # its longest member's (they start within a few us of each other).
            groups = []
            for label, us, gap in ds:
                if gap > 1e9 and groups:
                    groups[-1].append((label, us, None))
                else:
                    groups.append([(label, us, gap)])
            tot_us = sum(max(x[1] for x in g) + g[0][2] for g in groups)
            tot_gap = sum(g[0][2] for g in groups)
            tot_b = sum(bytes_of(x[0]) for x in ds)
            print(f"\n== layer {layer}: {tot_us:.1f} us ({tot_gap:.1f} of gaps, {len(groups)} barriers), {tot_b / 1e6:.1f} MB "
                  f"({tot_b / tot_us / 1e3 if tot_us else 0:.0f} GB/s over the layer)")
            print(f"{'dispatch':44s} {'us':>7s} {'gap':>5s} {'MB':>6s} {'GB/s':>5s} {'code':>6s}" +
                  (f" | {'other us':>8s} {'gap':>5s}" if ol else ''))
            od = ol[si][li][1] if ol else None
            for i, (label, us, gap) in enumerate(ds):
                gap = gap if gap < 1e9 else float('nan')
                b = bytes_of(label)
                name = label.split()[0]
                c = code.get(name.replace(' #2', ''), '')
                row = f"{label[:44]:44s} {us:7.1f} {gap:5.1f} {b / 1e6:6.2f} {b / us / 1e3 if us and b else 0:5.0f} {c!s:>6s}"
                if od and i < len(od):
                    row += f" | {od[i][1]:8.1f} {od[i][2]:5.1f}  {od[i][0][:30]}"
                print(row)


main()
