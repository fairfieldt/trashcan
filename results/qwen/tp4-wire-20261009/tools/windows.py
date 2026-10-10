#!/usr/bin/env python3
"""Per repetition of each run: the generation rate and incite's exchange per token, then every card's clock samples
inside the repetition's generation window (`tg_window_unix`): the share of busy samples (sclk above 300 MHz) below
850 MHz and the hottest reading. The clock logs come from scripts/with-clocks.sh on each Mac.
usage: windows.py INCITE_DIR INVITE_DIR PREFIX      (the pulled results/ of each Mac)"""
import json, os, sys

incite, invite, prefix = sys.argv[1:4]


def samples(path):
    out = []
    if os.path.exists(path):
        for line in list(open(path))[1:]:
            f = line.split()
            if len(f) == 7:
                out.append([float(f[0])] + [int(x) for x in f[1:]])
    return out


def card(rows, i, a, b):
    inside = [r for r in rows if a <= r[0] <= b]
    busy = [r for r in inside if r[1 + 3 * i] > 300]
    slow = sum(1 for r in busy if r[1 + 3 * i] < 850)
    hot = max((r[3 + 3 * i] for r in inside), default=0)
    return f'{slow}/{len(busy)} slow, {hot} C'


order = [l.split()[1] for l in open(os.path.join(incite, 'commands.log')) if l.split()[1].startswith(prefix)]
print('| run | rep | tok/s | exchange ms/token | incite card0 | incite card1 | invite card0 | invite card1 |')
print('|---|---|---|---|---|---|---|---|')
for name in order:
    mine = samples(os.path.join(incite, 'clocks', name + '.txt'))
    node = samples(os.path.join(invite, 'clocks', f'node-{name}-invite.txt'))
    for line in open(os.path.join(incite, name + '.json')):
        if not line.startswith('{'):
            continue
        r = json.loads(line)
        if 'tg_window_unix' not in r:
            continue
        a, b = r['tg_window_unix']
        print(f"| {name} | {r['rep']} | {r['tg_tok_s']:.2f} | {r['tg_tp_ms']['exchange']:.2f} | {card(mine, 0, a, b)} | "
              f"{card(mine, 1, a, b)} | {card(node, 0, a, b)} | {card(node, 1, a, b)} |")
