#!/usr/bin/env python3
"""Install the workstation netconsole receiver as a restarting macOS LaunchAgent."""
import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log', default=str(Path.home() / 'dev/claude-work/trashcan-kbuild/netconsole-incite.log'))
args = parser.parse_args()
directory = Path.home() / '.local/libexec'
directory.mkdir(parents=True, exist_ok=True)
receiver = directory / 'trashcan-netconsole-listen.py'
receiver.write_bytes(Path(__file__).with_name('netconsole-listen.py').read_bytes())
agent = Path.home() / 'Library/LaunchAgents/local.trashcan.netconsole.plist'
agent.parent.mkdir(parents=True, exist_ok=True)
agent.write_bytes(plistlib.dumps({
    'Label': 'local.trashcan.netconsole',
    'ProgramArguments': [sys.executable, str(receiver), str(Path(args.log).expanduser().resolve())],
    'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 3,
    'StandardErrorPath': str(directory / 'netconsole-errors.log'),
    'StandardOutPath': str(directory / 'netconsole-service.log'),
}))
domain = f'gui/{os.getuid()}'
subprocess.run(['launchctl', 'bootout', domain, str(agent)], capture_output=True)
subprocess.run(['launchctl', 'bootstrap', domain, str(agent)], check=True)
print(agent)
