#!/usr/bin/python3
# Convert Apple's .smc text firmware (D:addr:len:hex:sum and + continuation lines) to a flat binary.
import sys
img = bytearray(b'\xff' * 0x40000)
addr = None
for line in open(sys.argv[1]):
    f = [x.strip() for x in line.strip().split(':')]
    if f[0] == 'D':
        addr = int(f[1], 16); data = bytes.fromhex(f[3])
    elif f[0] == '+' and addr is not None:
        data = bytes.fromhex(f[2])
    else:
        addr = None if f[0] == 'H' else addr
        continue
    img[addr:addr + len(data)] = data
    addr += len(data)
open(sys.argv[2], 'wb').write(img)
