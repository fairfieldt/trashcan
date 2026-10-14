#!/bin/bash
# Kernel checks on one Mac Pro for the 7.2.9-p2p -> 7.3 A/B: the checks of results/kernel/linux-7.2-20261006/run.sh
# (system state, VBIOS per card, xgpu p2p-verify x3, p2p-wall, scripts/test-engine.sh) without its Gemma pp512 benches
# (their pinned binary is only on incite), plus a Flash-Next stage benchmark, forcing DPM high and back, and the amdgpu
# job timeout. Thunderbolt and netconsole are in tb.sh.
# usage, under the Mac's gpuq lease, from a workspace holding this script and scripts/ (clockmon.sh, clocksummary.sh,
# with-clocks.sh, latency-run.py):
#   ssh MAC 'cd ~/trashcan-exp/kernel73 && gpuq run ID -- ./run.sh [TAG]'    -> ./HOST-$(uname -r)[-TAG]/
# Uses ~/trashcan (bench/xgpu/xgpu, scripts/test-engine.sh and its release build) and the Flash-Next binary
# ~/trashcan-exp/flashnext/$FN_BIN (default trashcan-15e5457, main's binary of the 2026-10-14 main suite).
set -uo pipefail
cd "$(dirname "$0")"
here=$PWD
R=$here/$(hostname)-$(uname -r)${1:+-$1}
FN_BIN=${FN_BIN:-trashcan-15e5457}
FN=$HOME/trashcan-exp/flashnext
M=$HOME/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
mkdir -p "$R/clocks"
log() { echo "[$(date +%T)] $*"; }

log "system state"
{
    uname -a; cat /proc/cmdline; uptime
    dpkg-query -W -f '${Package} ${Version}\n' "linux-image-$(uname -r)"
    grep -E "^CONFIG_(PCI_P2PDMA|DRM_AMDGPU|DRM_AMDGPU_SI|DRM_AMD_DC_SI|DRM_VGEM|NETCONSOLE|USB4|USB4_NET|USB4_STREAM)=" \
        "/boot/config-$(uname -r)"
    sha256sum ~/trashcan/bench/xgpu/xgpu ~/trashcan/target/release/trashcan "$FN/$FN_BIN"
    for d in 02:00.0 06:00.0; do
        p=/sys/bus/pci/devices/0000:$d
        echo "$d $(sudo -n lspci -s $d -vv | grep -E 'LnkSta:' | tr -s ' \t' ' ')"
        echo "$d vbios $(cat $p/vbios_version) drm $(ls $p/drm | xargs) runtime_status $(cat $p/power/runtime_status)"
    done
    echo "amdgpu lockup_timeout='$(cat /sys/module/amdgpu/parameters/lockup_timeout)' (empty: driver default)"
    for p in si_support dc dpm; do echo "amdgpu $p=$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null)"; done
    lsmod | grep -E "^(amdgpu|radeon|vgem|netconsole|thunderbolt|thunderbolt_net|thunderbolt_stream) " | awk '{print "module", $1}'
} > "$R/system.txt" 2>&1
sudo -n dmesg > "$R/dmesg-full.txt" 2>&1
grep -i -E "amdgpu|radeon|vfct|vbios|p2p|drm" "$R/dmesg-full.txt" > "$R/dmesg-gpu.txt"
grep -E "WARNING|BUG|Oops|Call Trace|[Ee]rror|[Ff]ail|taint" "$R/dmesg-full.txt" > "$R/dmesg-warn.txt"

log "DPM: force high on each card, then auto"
{
    for p in /sys/bus/pci/drivers/amdgpu/0000:*; do
        h=$(echo $p/hwmon/hwmon*)
        clk() { echo "sclk $(($(cat $h/freq1_input) / 1000000)) mclk $(($(cat $h/freq2_input) / 1000000))"; }
        echo "$(basename $p) level $(cat $p/power_dpm_force_performance_level), $(clk)"
        echo high | sudo -n tee $p/power_dpm_force_performance_level > /dev/null; echo "  echo high: rc $?"
        sleep 2
        echo "  level $(cat $p/power_dpm_force_performance_level), $(clk)"
        echo auto | sudo -n tee $p/power_dpm_force_performance_level > /dev/null; echo "  echo auto: rc $?"
        sleep 2
        echo "  level $(cat $p/power_dpm_force_performance_level), $(clk)"
    done
} > "$R/dpm.txt" 2>&1
cat "$R/dpm.txt"

log "xgpu p2p-verify x3, p2p-wall"
scripts/clockmon.sh "$R/clocks/xgpu.txt" & mon=$!
cd ~/trashcan/bench/xgpu
for i in 1 2 3; do
    ./xgpu p2p-verify > "$R/xgpu-verify-$i.txt" 2>&1; echo "exit $?" >> "$R/xgpu-verify-$i.txt"
    tail -2 "$R/xgpu-verify-$i.txt"
done
./xgpu p2p-wall > "$R/xgpu-p2p-wall.txt" 2>&1; echo "exit $?" >> "$R/xgpu-p2p-wall.txt"
cd "$here"
kill $mon; wait $mon 2>/dev/null
scripts/clocksummary.sh "$R/clocks/xgpu.txt"

log "Flash-Next stage-bench ($FN_BIN): one warm-up, three runs"
for i in 0 1 2 3; do
    scripts/with-clocks.sh "$R/clocks/sb-$i.txt" sudo -n python3 scripts/latency-run.py "$FN/$FN_BIN" qwen4 stage-bench "$M" \
        --ranges 0:4,4:8 --gpus 0,1 --context 32768 --head-rows 0:21504,21504:43008 --batch 128 \
        --prefill 8 --decode 32 --start 64 > "$R/sb-$i.json" 2> "$R/sb-$i.err"
    echo "sb-$i rc $? $(python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print('chunk of 128: %.2f ms wall, decode step %.2f ms wall' % (r['prefill_wall_ms'], r['decode_wall_ms']))
" "$R/sb-$i.json" 2>&1) $(tail -1 "$R/sb-$i.err")"
done

log "scripts/test-engine.sh (in ~/trashcan)"
scripts/clockmon.sh "$R/clocks/test-engine.txt" & mon=$!
(cd ~/trashcan && scripts/test-engine.sh) > "$R/test-engine.log" 2>&1
echo "test-engine exit $?, $(grep -c '^PASS' "$R/test-engine.log") validations PASS, $(grep -E '^test result' "$R/test-engine.log")"
kill $mon; wait $mon 2>/dev/null
scripts/clocksummary.sh "$R/clocks/test-engine.txt"
grep "worst relative error" "$R/test-engine.log"
log "done: $R"
