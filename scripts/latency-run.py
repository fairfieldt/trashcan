#!/usr/bin/env python3
"""Hold a CPU PM QoS latency request only for a command's lifetime (user authorized)."""
import os
import pwd
import struct
import subprocess
import sys

if len(sys.argv) < 2:
    sys.exit("usage: sudo -n python3 scripts/latency-run.py COMMAND [ARG ...]")
with open("/dev/cpu_dma_latency", "wb", buffering=0) as request:
    request.write(struct.pack("=i", 0))
    # Only the open requires privilege. Run the benchmark as the invoking user.
    command_env = os.environ.copy()
    if os.geteuid() == 0 and "SUDO_UID" in os.environ:
        uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
        user = pwd.getpwuid(uid)
        os.initgroups(user.pw_name, gid)
        os.setgid(gid)
        os.setuid(uid)
        command_env.update(HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name)
    print("CPU latency request: 0 us, scoped to this command", file=sys.stderr)
    result = subprocess.run(sys.argv[1:], check=False, env=command_env)
sys.exit(result.returncode)
