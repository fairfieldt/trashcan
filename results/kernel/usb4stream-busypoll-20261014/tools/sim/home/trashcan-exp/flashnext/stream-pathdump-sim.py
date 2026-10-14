#!/usr/bin/env python3
"""Made-up path tables of a simulated Mac in pathdump.sh's format: two controllers (0-0, 1-0), lane adapters 1-4 and
the NHI adapter 5; the cable on port 3. thunderbolt_net's paths are always up (Tx ring 2 to HopID 8, Rx HopID 8 to
ring 2); the stream's (HopID 15 both ways, ring 3) are up while the holdopen stub runs. On alight the two came up the
other way round (stream ring 2, thunderbolt_net ring 3)."""
import os, subprocess, time

host = os.environ.get('SIMHOST', '?')
held = os.path.exists(os.path.join(os.environ['HOME'], 'holding'))
print(f'# host {host} {subprocess.run(["uname", "-r"], capture_output=True, text=True).stdout.strip()} {time.strftime("%T")}')
links = subprocess.run(['tblink', 'links'], capture_output=True, text=True).stdout
for l in links.splitlines():
    print(f'# links {l}')
for dom in (0, 1):
    print(f'# group /sys/kernel/config/thunderbolt/stream/{dom}-3.1/trashcan busy_poll=0 in_hopid=15 index={dom} '
          f'out_hopid=15 ring_size=256')
print('# kernel [    5.1] thunderbolt 0-3: new host found, vendor=0x8086 device=0x1')


def d0(nxt, port, credits, en):
    return en << 31 | credits << 17 | port << 11 | nxt


def d1(ifc, efc):
    return 0x7ff << 12 | efc << 25 | ifc << 24 | 5 << 8 | 1


tb, st = (3, 2) if host == 'alight' else (2, 3)
for dom in (0, 1):
    r = f'{dom}-0'
    for off in range(8):
        print(f'{r} - swregs 0x{off:04x} {off:4d} 0x00 0x00 0x{0x1000 + off:08x}')
    for port in range(1, 6):
        typ = 2 if port == 5 else 1
        regs = [0x15cb8086, 0x0, typ | 1 << 24, port << 20, 16 << 20 | 2, 0x2, 0, 0]
        for off, v in enumerate(regs):
            print(f'{r} port{port} regs 0x{off:04x} {off:4d} 0x00 0x00 0x{v:08x}')
        if typ == 1:
            print(f'{r} port{port} regs 0x0013    0 0x01 0x00 0x{0x0230000:08x}')
            print(f'{r} port{port} regs 0x0014    1 0x01 0x00 0x{(0x2 << 20 | 0x8 << 16 | (1 << 15 if port == 3 else 0)):08x}')
        path = {0: (d0(port, 0, 2, 1), 0x21000001)}
        if port == 5:
            path[tb] = (d0(8, 3, 14, 1), d1(1, 1))
            path[st] = (d0(15, 3, 14, int(held)), d1(int(held), int(held)))
        elif port == 3:
            path[8] = (d0(tb, 5, 14, 1), d1(1, 0))
            path[15] = (d0(st, 5, 14, int(held)), d1(int(held), 0))
        for hop in range(0 if port != 5 else 0, 16 if port != 5 else 12):
            if port != 5 and 0 < hop < 8:
                continue
            a, b = path.get(hop, (0, 0x22000001))
            print(f'{r} port{port} path 0x{2 * hop:04x}    0 0x{hop:02x} 0x{a:08x}')
            print(f'{r} port{port} path 0x{2 * hop + 1:04x}    1 0x{hop:02x} 0x{b:08x}')
print(f'# end {time.strftime("%T")}')
