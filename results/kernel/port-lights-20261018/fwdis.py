#!/usr/bin/env python3
# Thumb-2 helpers for the Mac Pro 6,1 SMC firmware image (fw.bin, loaded at 0).
#   fwdis.py dis ADDR [COUNT]      disassemble, resolving literal loads and naming key variables/handlers
#   fwdis.py func ADDR             disassemble until the first return-like instruction after ADDR
#   fwdis.py lit VALUE [VALUE2]    code that loads a literal equal to VALUE (or in [VALUE, VALUE2])
#   fwdis.py calls ADDR            BL/B.W call sites targeting ADDR
import struct, sys
from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_MCLASS
from capstone.arm import ARM_OP_MEM, ARM_REG_PC

import os; d = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fw.bin'), 'rb').read()
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
md.detail = True

names = {}
for off in range(0x2f000, 0x34000, 16):
    e = d[off:off + 16]
    if not all(32 <= c < 127 for c in e[:4]):
        break
    k = e[:4].decode()
    ptr = struct.unpack('<I', e[12:16])[0]
    names.setdefault(ptr & ~1, k)

def name(v):
    for delta in range(0, 4):
        if v - delta in names:
            return names[v - delta] + (f'+{delta}' if delta else '')
    return ''

def lit(insn):
    # value loaded by ldr rX, [pc, #imm]
    if insn.mnemonic.startswith('ldr') and len(insn.operands) == 2:
        op = insn.operands[1]
        if op.type == ARM_OP_MEM and op.mem.base == ARM_REG_PC:
            a = ((insn.address + 4) & ~3) + op.mem.disp
            if 0 <= a < len(d) - 4:
                return a, struct.unpack('<I', d[a:a + 4])[0]
    return None

def show(insn):
    extra = ''
    l = lit(insn)
    if l:
        extra = f'  ; ={l[1]:#x} {name(l[1])}'
    elif insn.mnemonic in ('bl', 'b.w', 'b', 'blx') and insn.op_str.startswith('#'):
        t = int(insn.op_str[1:], 16)
        extra = f'  ; {name(t)}' if name(t) else ''
    print(f'{insn.address:#07x}: {insn.bytes.hex():12} {insn.mnemonic} {insn.op_str}{extra}')

def walk(addr, count):
    n = 0
    while n < count:
        got = False
        for insn in md.disasm(d[addr:addr + 64], addr):
            yield insn
            addr = insn.address + insn.size
            n += 1
            got = True
            if n >= count:
                return
        if not got:
            print(f'{addr:#07x}: .short {struct.unpack("<H", d[addr:addr+2])[0]:#06x}')
            addr += 2
            n += 1

cmd = sys.argv[1]
if cmd == 'dis':
    for insn in walk(int(sys.argv[2], 16) & ~1, int(sys.argv[3]) if len(sys.argv) > 3 else 40):
        show(insn)
elif cmd == 'func':
    for insn in walk(int(sys.argv[2], 16) & ~1, 400):
        show(insn)
        s = insn.mnemonic + ' ' + insn.op_str
        if (insn.mnemonic.startswith('pop') and 'pc' in s) or s.strip() == 'bx lr' or (insn.mnemonic.startswith('ldm') and 'pc' in s):
            break
elif cmd in ('lit', 'calls'):
    lo = int(sys.argv[2], 16)
    hi = int(sys.argv[3], 16) if len(sys.argv) > 3 else lo
    # linear sweep of all code regions at both halfword alignments is noisy; sweep from every halfword
    seen = set()
    for start in range(0xb800, 0x2f000, 0x800):
        addr = start
        for insn in md.disasm(d[start:start + 0x800 + 8], start):
            if insn.address >= start + 0x800:
                break
            if cmd == 'lit':
                l = lit(insn)
                if l and lo <= l[1] <= hi and insn.address not in seen:
                    seen.add(insn.address)
                    show(insn)
            elif insn.mnemonic in ('bl', 'b.w', 'blx') and insn.op_str.startswith('#'):
                if lo <= int(insn.op_str[1:], 16) <= hi:
                    show(insn)
