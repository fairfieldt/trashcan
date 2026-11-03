#!/bin/sh
# Chooses what a test machine set up for network boot (README, "Network boot") boots next: "dross" chainloads
# target/netboot/dross.efi (build.sh with NETBOOT=1), "linux" leaves GRUB's own default. The machine's GRUB
# reads target/netboot/boot.cfg over HTTP at every boot.
#   ./netboot.sh dross [IMAGE.efi] | linux
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 - "$project_dir" "$@" <<'PY'
import argparse, hashlib, os, tempfile
from pathlib import Path
import sys

root = Path(sys.argv[1]) / 'target/netboot'
parser = argparse.ArgumentParser(description='Publish a complete image for the next network boot.')
parser.add_argument('mode', choices=['dross', 'linux'])
parser.add_argument('image', nargs='?', type=Path)
args = parser.parse_args(sys.argv[2:])
if args.mode == 'linux' and args.image:
    parser.error('linux takes no image')

def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

configuration = b''
if args.mode == 'dross':
    source = args.image or root / 'dross.efi'
    data = source.read_bytes()
    if not data.startswith(b'MZ'):
        parser.error(f'{source} is not an EFI executable')
    digest = hashlib.sha256(data).hexdigest()
    # A reader always sees one complete immutable build, even during publication.
    image = root / 'builds' / (digest + '.efi')
    publish(image, data)
    configuration = f'''menuentry "dross (network)" --id dross-net {{
    chainloader (http,${{dross_server}})/builds/{digest}.efi
}}
set default=dross-net
set timeout=1
'''.encode()
    print(f'image: {digest}')
publish(root / 'boot.cfg', configuration)
print(f'next boot: {args.mode}')
PY
