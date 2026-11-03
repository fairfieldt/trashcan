#!/usr/bin/env python3
"""Build a self-contained test loader without changing the live netboot tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parent


def run(command, **kwargs):
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--init-conf', required=True, type=Path)
    parser.add_argument('--rootfs', required=True, type=Path)
    parser.add_argument('--features', default='unix,audit-test')
    parser.add_argument('--esp', type=Path, help='Use an already tested ESP snapshot instead of rebuilding its kernel and native programs')
    parser.add_argument('--module', action='append', default=[], metavar='NAME=FILE')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'cargo'
    modules = out / 'modules'
    # A fresh manifest selects files; previous stage files cannot enter the image.
    modules.mkdir(exist_ok=True)
    features = ['--features', args.features] if args.features else []
    if not args.esp:
        run(['cargo', 'build', '-p', 'dross', '--release', '--target', 'x86_64-unknown-none',
             '--target-dir', target, *features])
        run(['cargo', 'build', '-p', 'init', '-p', 'selftest', '-p', 'pcid', '-p', 'usbd',
             '--release', '--target', 'x86_64-unknown-none', '--target-dir', target])
    inputs = {'dross.elf': args.esp.resolve() / 'dross.elf' if args.esp else target / 'x86_64-unknown-none/release/dross',
              'rootfs.tar': args.rootfs.resolve(), 'init.conf': args.init_conf.resolve()}
    for name in ['init', 'selftest', 'pcid', 'usbd']:
        inputs[name] = args.esp.resolve() / 'modules' / name if args.esp else target / 'x86_64-unknown-none/release' / name
    supplied = set()
    for item in args.module:
        name, separator, path = item.partition('=')
        if not separator or not name or '/' in name or len(name.encode()) > 31:
            parser.error('module must be NAME=FILE with a name of at most 31 bytes')
        if name in supplied or name in {'dross.elf', 'rootfs.tar', 'init.conf'}:
            parser.error(f'duplicate module: {name}')
        supplied.add(name)
        inputs[name] = Path(path).resolve()
    if len(inputs) - 1 > 64:
        parser.error('too many modules')
    manifest = {}
    # Stripped test copies keep the ESP artifact compact. Originals stay intact.
    sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
    objcopy = next(sysroot.glob('lib/rustlib/*/bin/llvm-objcopy'))
    staged = {}
    for name, source in sorted(inputs.items()):
        destination = modules / name
        shutil.copy2(source, destination)
        if destination.read_bytes()[:4] == b'\x7fELF':
            run([objcopy, '--strip-all', destination])
        staged[name] = destination
    bundle = out / 'bundle'
    with bundle.open('wb') as stream:
        stream.write(b'DROSSBND')
        for name, path in sorted(staged.items()):
            data = path.read_bytes()
            encoded = name.encode()
            stream.write(struct.pack('<H', len(encoded)) + encoded + struct.pack('<Q', len(data)) + data)
            manifest[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        stream.write(struct.pack('<H', 0))
    run(['cargo', 'build', '-p', 'dross-boot', '--release', '--features', 'bundle',
         '--target', 'x86_64-unknown-uefi', '--target-dir', target],
        env={**os.environ, 'DROSS_BUNDLE': str(bundle)})
    loader = out / 'BOOTX64.EFI'
    shutil.copy2(target / 'x86_64-unknown-uefi/release/dross-boot.efi', loader)
    manifest['BOOTX64.EFI'] = {'bytes': loader.stat().st_size,
                              'sha256': hashlib.sha256(loader.read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(loader)


if __name__ == '__main__':
    main()
