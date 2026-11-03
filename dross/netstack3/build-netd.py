#!/usr/bin/env python3
"""Build the userspace network service with its pinned Netstack3 toolchain."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    if os.environ.get('RUSTFLAGS') or os.environ.get('CARGO_ENCODED_RUSTFLAGS'):
        raise SystemExit('Unset RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS to use the target config')
    subprocess.run([sys.executable, HERE / 'prepare.py'], cwd=HERE, check=True)
    build = HERE / 'target/service-build'
    environment = dict(os.environ, CARGO_TARGET_DIR=str(build))
    subprocess.run(['cargo', 'build', '--locked', '-p', 'oscar-netd',
                    '--target', 'x86_64-unknown-none', '--release'],
                   cwd=HERE, env=environment, check=True)
    destination = HERE.parent / 'unix/target/native/x86_64-unknown-none/release'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['unix-netd', 'native-net-test']:
        shutil.copy2(build / 'x86_64-unknown-none/release' / name, destination / name)


if __name__ == '__main__':
    main()
