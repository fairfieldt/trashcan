"""Overwrite tensors of a split GGUF in place with same-type, same-size data from files (no numpy).

usage: gguf_splice.py [--save DIR] FIRST_SHARD NAME=FILE [NAME=FILE ...]
--save DIR first copies each tensor's current bytes to DIR/NAME (so the splice can be undone).
Every shard of the set is scanned; each named tensor must already have the replacement's size (for example a
placeholder of the same type made by llama-quantize)."""
import glob
import os
import re
import struct
import sys

TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # fixed-size metadata value types
BLOCKS = {2: (32, 18), 8: (32, 34), 20: (32, 18), 23: (256, 136), 42: (64, 18)}  # ggml type -> (weights, bytes)


def read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')


def skip_value(f, vt):
    if vt in TYPE_SIZES:
        f.seek(TYPE_SIZES[vt], 1)
    elif vt == 8:  # string
        read_str(f)
    elif vt == 9:  # array
        (et, n) = struct.unpack('<IQ', f.read(12))
        if et in TYPE_SIZES:
            f.seek(TYPE_SIZES[et] * n, 1)
        else:
            for _ in range(n):
                skip_value(f, et)
    else:
        raise ValueError(f'unknown metadata type {vt}')


def tensors(path):
    """Name -> (type, absolute data offset) for one shard."""
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        (version, n_tensors, n_kv) = struct.unpack('<IQQ', f.read(20))
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            (vt,) = struct.unpack('<I', f.read(4))
            if key == 'general.alignment':
                (alignment,) = struct.unpack('<I', f.read(4))
            else:
                skip_value(f, vt)
        infos = []
        for _ in range(n_tensors):
            name = read_str(f)
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack(f'<{nd}Q', f.read(8 * nd))
            (ty, off) = struct.unpack('<IQ', f.read(12))
            n = 1
            for x in dims:
                n *= x
            size = n // BLOCKS[ty][0] * BLOCKS[ty][1] if ty in BLOCKS else None
            infos.append((name, ty, off, size))
        base = (f.tell() + alignment - 1) // alignment * alignment
    return {name: (ty, base + off, size) for name, ty, off, size in infos}


def main():
    args = sys.argv[1:]
    save = None
    if args[0] == '--save':
        save, args = args[1], args[2:]
        os.makedirs(save, exist_ok=True)
    first = args[0]
    shards = sorted(glob.glob(re.sub(r'-\d{5}-of-(\d{5})\.gguf$', r'-*-of-\1.gguf', first)))
    want = dict(a.split('=', 1) for a in args[1:])
    for shard in shards:
        index = tensors(shard)
        for name, src in list(want.items()):
            if name not in index:
                continue
            ty, off, cur = index[name]
            size = os.path.getsize(src)
            if cur != size:
                sys.exit(f'{name}: {cur} bytes of type {ty} in the GGUF, {size} in {src}')
            with open(shard, 'r+b') as f:
                if save:
                    f.seek(off)
                    with open(os.path.join(save, name), 'wb') as o:
                        left = size
                        while left:
                            chunk = f.read(min(left, 1 << 24))
                            o.write(chunk)
                            left -= len(chunk)
                f.seek(off)
                with open(src, 'rb') as s:
                    while chunk := s.read(1 << 24):
                        f.write(chunk)
            print(f'{name}: {size} bytes at {off} in {os.path.basename(shard)} (type {ty})')
            del want[name]
    if want:
        sys.exit(f'not found: {sorted(want)}')


if __name__ == '__main__':
    main()
