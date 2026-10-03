#!/usr/bin/env python3
"""Sample read-only D700 memory counters until the specified benchmark exits."""
import csv
from pathlib import Path
import sys
import time

pid = int(sys.argv[1])
out = Path(sys.argv[2])
fields = ['t'] + [f'{kind}{g}_bytes' for g in [0, 1] for kind in ['vram', 'vis_vram', 'gtt']]
with out.open('w') as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    deadline = time.monotonic() + 3600
    while Path(f'/proc/{pid}').exists() and time.monotonic() < deadline:
        row = {'t': time.time()}
        for g in [0, 1]:
            base = Path(f'/sys/class/drm/card{g}/device')
            for kind in ['vram', 'vis_vram', 'gtt']:
                row[f'{kind}{g}_bytes'] = int((base / f'mem_info_{kind}_used').read_text())
        writer.writerow(row)
        f.flush()
        time.sleep(1)
