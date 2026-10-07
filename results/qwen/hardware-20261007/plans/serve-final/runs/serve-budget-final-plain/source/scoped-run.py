#!/usr/bin/env python3
"""Run one benchmark with temporary performance governors and CPU latency=0.

This helper runs as root under gpuq, but demotes its child to the sudo caller.
It deliberately does not forward termination signals to GPU work.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import struct
import subprocess
import sys
import time


def fan_snapshot(hwmon_root=Path('/sys/class/hwmon'),
                 curve_path=Path('/usr/local/sbin/fancurve.sh'),
                 platform_root=Path('/sys/devices/platform')):
    """Best-effort provenance only: never change fan controls or fail the run."""
    snapshot = dict(unix=time.time(), hwmon=[], fancurve_path=str(curve_path),
                    fancurve_sha256=None, read_errors={})
    errors = snapshot['read_errors']
    attributes = ('fan1_min', 'fan1_max', 'fan1_manual', 'fan1_output', 'fan1_input')

    def read(path):
        try:
            return path.read_text().strip()
        except Exception as error:
            errors[str(path)] = repr(error)
            return None

    try:
        snapshot['fancurve_sha256'] = hashlib.sha256(curve_path.read_bytes()).hexdigest()
    except Exception as error:
        errors[str(curve_path)] = repr(error)
    devices = []
    try:
        for device in sorted(hwmon_root.iterdir()):
            devices.append(device)
            # Older applesmc drivers expose attributes on hwmonN/device.
            if (device / 'device').exists():
                devices.append(device / 'device')
    except Exception as error:
        errors[str(hwmon_root)] = repr(error)
    try:
        devices.extend(path for path in sorted(platform_root.iterdir())
                       if path.name.startswith('applesmc.'))
    except Exception as error:
        errors[str(platform_root)] = repr(error)
    seen = set()
    for device in devices:
        try:
            resolved = str(device.resolve())
            if resolved in seen:
                continue
            entries = {path.name for path in device.iterdir()}
            if not entries.intersection(attributes):
                continue
            seen.add(resolved)
            row = dict(path=str(device), device_path=resolved, name=read(device / 'name'))
            row.update({attribute: read(device / attribute) for attribute in attributes})
            snapshot['hwmon'].append(row)
        except Exception as error:
            errors[str(device)] = repr(error)
    if not snapshot['hwmon']:
        errors.setdefault(str(hwmon_root), 'no fan1 attributes found')
    return snapshot


def main():
    state_path, *command = sys.argv[1:]
    if not command or os.geteuid() != 0 or 'SUDO_UID' not in os.environ:
        raise SystemExit('run via sudo -n; usage: scoped-run.py STATE COMMAND ...')
    # An SSH disconnect or an interrupted local harness must not interrupt a
    # queue submission with a GPU timeline/event gate still pending.
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)
    uid, gid = int(os.environ['SUDO_UID']), int(os.environ['SUDO_GID'])
    user = pwd.getpwuid(uid)
    paths = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))
    if not paths:
        paths = sorted(Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor'))
    if not paths:
        raise SystemExit('no CPU governors available')
    before = {str(path): path.read_text().strip() for path in paths}
    state = dict(command=command, wrapper_pid=os.getpid(), started_unix=time.time(),
                 governors_before=before, cpu_latency_us=0,
                 binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                 kernel=Path('/proc/sys/kernel/osrelease').read_text().strip(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

    def save():
        temporary = Path(state_path + '.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.chown(temporary, uid, gid)
        temporary.replace(state_path)

    def demote():
        # Popen does not reset signals inherited as SIG_IGN.
        for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)
        os.initgroups(user.pw_name, gid)
        os.setgid(gid)
        os.setuid(uid)

    save()
    try:
        state['fan_before'] = fan_snapshot()
        save()
        for path in paths:
            path.write_text('performance')
        state['governors_during'] = {str(path): path.read_text().strip() for path in paths}
        if any(value != 'performance' for value in state['governors_during'].values()):
            raise RuntimeError('performance governor did not take effect')
        with open('/dev/cpu_dma_latency', 'wb', buffering=0) as request:
            request.write(struct.pack('=i', 0))
            env = os.environ.copy()
            env.update(HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name,
                       TRASHCAN_CPU_LATENCY_US='0')
            child = subprocess.Popen(command, env=env, preexec_fn=demote,
                                     start_new_session=True)
            state.update(pid=child.pid, process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
            save()
            code = child.wait()
            state.update(returncode=code, finished_unix=time.time())
    finally:
        state['fan_after'] = fan_snapshot()
        errors = {}
        for path, value in before.items():
            try:
                Path(path).write_text(value)
            except OSError as error:
                errors[path] = str(error)
        state['governors_after'] = {str(path): path.read_text().strip() for path in paths}
        state['restore_errors'] = errors
        save()
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    sys.exit(main())
