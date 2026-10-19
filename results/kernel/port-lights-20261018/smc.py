#!/usr/bin/python3
# Raw Apple SMC access through /dev/port (the applesmc protocol, ports 0x300/0x304).
# Unload applesmc first (rmmod applesmc) so nothing else talks to the SMC at the same time.
#   smc.py list [PREFIX]          every key: name, type, length, flags, value
#   smc.py info KEY               type, length, flags
#   smc.py read KEY               value as hex
#   smc.py write KEY HEX          write HEX bytes (length must match the key's)
import os, sys, time

DATA, CMD = 0x300, 0x304
READ, WRITE, BY_INDEX, TYPE = 0x10, 0x11, 0x12, 0x13
AWAITING, IB_CLOSED, BUSY = 1, 2, 4
fd = os.open('/dev/port', os.O_RDWR)

def inb(p): return os.pread(fd, 1, p)[0]
def outb(v, p): os.pwrite(fd, bytes([v]), p)

def wait(val, mask):
    us = 8
    for i in range(24):
        if inb(CMD) & mask == val:
            return
        time.sleep(us / 1e6)
        if i > 9:
            us <<= 1
    raise IOError(f'SMC status {inb(CMD):#x} never matched {val:#x}/{mask:#x}')

def send_byte(b, port):
    wait(0, IB_CLOSED)
    wait(BUSY, BUSY)
    outb(b, port)

def send_command(c):
    wait(0, IB_CLOSED)
    outb(c, CMD)

def sane():
    try:
        wait(0, BUSY)
    except IOError:
        send_command(READ)
        wait(0, BUSY)

def transact_read(cmd, key, n):
    sane()
    send_command(cmd)
    for b in key:
        send_byte(b, DATA)
    send_byte(n, DATA)
    out = bytearray()
    for _ in range(n):
        wait(AWAITING | BUSY, AWAITING | BUSY)
        out.append(inb(DATA))
    for _ in range(16):
        time.sleep(8e-6)
        if not inb(CMD) & AWAITING:
            break
        inb(DATA)
    wait(0, BUSY)
    return bytes(out)

def transact_write(key, data):
    sane()
    send_command(WRITE)
    for b in key:
        send_byte(b, DATA)
    send_byte(len(data), DATA)
    for b in data:
        send_byte(b, DATA)
    wait(0, BUSY)

def info(key):
    i = transact_read(TYPE, key.encode(), 6)
    return i[0], i[1:5].decode('latin1'), i[5]

def read(key):
    n, _, _ = info(key)
    return transact_read(READ, key.encode(), n)

def fmt(key):
    try:
        n, t, f = info(key)
    except IOError as e:
        return f'{key} info failed: {e}'
    try:
        v = transact_read(READ, key.encode(), n).hex(' ') if f & 0x80 else '(not readable)'
    except IOError as e:
        v = f'(read failed: {e})'
    return f'{key} [{t}] len {n} flags {f:#04x} {v}'

def main():
    cmd = sys.argv[1]
    if len(sys.argv) > 2 and cmd != 'list' and cmd != 'watch' and len(sys.argv[2]) != 4:
        sys.exit(f'key must be 4 characters: {sys.argv[2]!r}')
    if cmd == 'list':
        prefix = sys.argv[2] if len(sys.argv) > 2 else ''
        count = int.from_bytes(transact_read(READ, b'#KEY', 4), 'big')
        for i in range(count):
            key = transact_read(BY_INDEX, i.to_bytes(4, 'big'), 4).decode('latin1')
            if key.startswith(prefix):
                print(fmt(key), flush=True)
    elif cmd == 'watch':
        # print every change of the readable keys starting with any of PREFIXES (default L)
        prefixes = tuple(sys.argv[2:]) or ('L',)
        count = int.from_bytes(transact_read(READ, b'#KEY', 4), 'big')
        keys = []
        for i in range(count):
            key = transact_read(BY_INDEX, i.to_bytes(4, 'big'), 4).decode('latin1')
            if key.startswith(prefixes):
                n, _, f = info(key)
                if f & 0x80:
                    keys.append((key, n))
        last, t0 = {}, time.time()
        print(f'watching {len(keys)} keys', flush=True)
        while True:
            for key, n in keys:
                try:
                    v = transact_read(READ, key.encode(), n).hex(' ')
                except IOError:
                    continue
                if key in last and last[key] != v:
                    print(f'{time.time() - t0:7.2f} {key} {last[key]} -> {v}', flush=True)
                last[key] = v
    elif cmd == 'probe':
        # probe KEY HEX [SECS]: write, then print each change of LsbR (the lights' level) for SECS (default 3)
        key, data = sys.argv[2], bytes.fromhex(sys.argv[3])
        secs = float(sys.argv[4]) if len(sys.argv) > 4 else 3
        before = transact_read(READ, b'LsbR', 3).hex(' ')
        transact_write(key.encode(), data)
        t0, last = time.time(), before
        print(f'{key}={data.hex(" ")}: LsbR {before}', end='', flush=True)
        while time.time() - t0 < secs:
            v = transact_read(READ, b'LsbR', 3).hex(' ')
            if v != last:
                print(f' | {time.time() - t0:.2f}s {v}', end='', flush=True)
                last = v
        print()
    elif cmd == 'info':
        print(fmt(sys.argv[2]))
    elif cmd == 'read':
        print(read(sys.argv[2]).hex(' '))
    elif cmd == 'write':
        transact_write(sys.argv[2].encode(), bytes.fromhex(sys.argv[3]))
        print(fmt(sys.argv[2]))
    else:
        sys.exit(__doc__)

main()
