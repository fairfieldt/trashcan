"""Read-only facts about a trashcan node, as JSON, for deploy/trashcan-cluster (sent over SSH: python3 - REQUEST).

The only file it writes is its sha256 cache (~/.cache/trashcan-cluster/sha256.json), keyed by path, size, mtime and
inode, so a file is hashed once until it changes.
"""
import glob
import grp
import hashlib
import json
import os
import pwd
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

req = json.loads(sys.argv[1])
home = os.path.expanduser("~")
CACHE = os.path.join(home, ".cache/trashcan-cluster/sha256.json")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def expand(p):
    return os.path.expanduser(p)


try:
    with open(CACHE) as f:
        cache = json.load(f)
except (OSError, ValueError):
    cache = {}


def sha256(path):
    st = os.stat(path)
    key = os.path.realpath(path)
    stamp = [st.st_size, st.st_mtime_ns, st.st_ino]
    hit = cache.get(key)
    if hit and hit[:3] == stamp:
        return hit[3]
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    cache[key] = [*stamp, h.hexdigest()]
    return cache[key][3]


def file_info(path, do_hash):
    p = expand(path)
    if not os.path.isfile(p):
        return path, {"exists": False}
    info = {"exists": True, "size": os.path.getsize(p), "real": os.path.realpath(p)}
    if do_hash:
        info["sha256"] = sha256(p)
    return path, info


out = {"hostname": sh("hostname"), "user": pwd.getpwuid(os.getuid()).pw_name, "home": home}
rel = os.uname().release
out["kernel"] = rel
out["cmdline"] = read("/proc/cmdline")
# The patch set id that kernel/build-7.3.sh puts in the package version (+psXXXXXXXX), if any.
ver = sh(f"dpkg-query -W -f '${{Version}}' linux-image-{rel} 2>/dev/null")
out["kernel_package_version"] = ver
m = re.search(r"\+ps([0-9a-f]{8})", ver)
out["kernel_patchset"] = m.group(1) if m else None
cfg = read(f"/boot/config-{rel}")
out["p2pdma"] = "CONFIG_PCI_P2PDMA=y" in cfg.splitlines()
norm = "\n".join(l for l in cfg.splitlines() if not l.startswith("CONFIG_LOCALVERSION="))
out["boot_config_sha256"] = hashlib.sha256(norm.encode()).hexdigest() if cfg else None
out["mesa"] = sh("dpkg-query -W -f '${Version}' mesa-vulkan-drivers 2>/dev/null")

gpus = []
for d in sorted(glob.glob("/sys/bus/pci/drivers/amdgpu/0000:*")):
    card = [c for c in glob.glob(f"{d}/drm/card*") if re.fullmatch(r"card\d+", os.path.basename(c))]
    temp = glob.glob(f"{d}/hwmon/hwmon*/temp1_input")
    gpus.append({
        "pci": os.path.basename(d),
        "vbios": read(f"{d}/vbios_version"),
        "vram_total_mib": int(read(f"{d}/mem_info_vram_total", "0")) >> 20,
        "vram_used_mib": int(read(f"{d}/mem_info_vram_used", "0")) >> 20,
        "dpm": read(f"{d}/power_dpm_force_performance_level"),
        "temp_c": int(read(temp[0], "0")) // 1000 if temp else None,
        "card": os.path.basename(card[0]) if card else None,
    })
out["gpus"] = gpus
try:
    render = grp.getgrnam("render")
    out["in_render"] = render.gr_gid in os.getgroups() or out["user"] in render.gr_mem
except KeyError:
    out["in_render"] = False
out["services"] = {s: sh(f"systemctl is-active {s}.service") for s in ("fancurve", "tblink")}
mounts = read("/proc/mounts")
out["configfs"] = " /sys/kernel/config configfs " in mounts
out["debugfs"] = " /sys/kernel/debug debugfs " in mounts
out["thunderbolt_stream"] = os.path.isdir("/sys/module/thunderbolt_stream")
out["streams"] = {os.path.basename(s): os.access(s, os.R_OK | os.W_OK) for s in glob.glob("/dev/tbstream*")}
out["engines"] = [l for l in sh(r"pgrep -af '(^|/)trashcan[^ /]* (serve|server|chat|bench|validate|qwen|qwen4|qwen-[a-z-]+) '").splitlines()
                  if l]
busy = []
for port in req.get("ports", []):
    if sh(f"ss -ltnH 'sport = :{port}'"):
        busy.append(port)
out["busy_ports"] = busy

files = list(req.get("files", []))
if req.get("binary"):
    files.append(req["binary"])
with ThreadPoolExecutor(max_workers=4) as pool:
    results = dict(pool.map(lambda p: file_info(p, req.get("hash")), files))
if req.get("binary"):
    out["binary"] = results.pop(req["binary"])
out["files"] = results

if req.get("hash"):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE)
print(json.dumps(out))
