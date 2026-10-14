#!/bin/bash
# Short check of one Mac Pro after booting a kernel, for the six-Mac rollout (run.sh is the full A/B): system state and
# VBIOS per card, the fan service and fan speed, DPM high then auto on both cards, one xgpu p2p-verify, and dmesg's
# warnings. p2p-verify uses the GPUs, so run it under the Mac's gpuq lease, from a workspace holding this script and
# scripts/ (clockmon.sh, clocksummary.sh):
#   ssh MAC 'cd ~/trashcan-exp/kernel73 && gpuq run ID -- ./check.sh'    -> ./HOST-$(uname -r)-check/
set -uo pipefail
cd "$(dirname "$0")"
here=$PWD
R=$here/$(hostname)-$(uname -r)-check
mkdir -p "$R/clocks"
{
    uname -a; cat /proc/cmdline; uptime
    dpkg-query -W -f '${Package} ${Version}\n' "linux-image-$(uname -r)"
    grep "^GRUB_DEFAULT=" /etc/default/grub
    for d in 02:00.0 06:00.0; do
        p=/sys/bus/pci/devices/0000:$d
        echo "$d $(sudo -n lspci -s $d -vv | grep -E 'LnkSta:' | tr -s ' \t' ' ')"
        echo "$d vbios $(cat $p/vbios_version) runtime_status $(cat $p/power/runtime_status)"
    done
    echo "amdgpu lockup_timeout='$(cat /sys/module/amdgpu/parameters/lockup_timeout)' (empty: driver default)"
    smc=/sys/devices/platform/applesmc.768
    fan=$smc; [ -e $smc/fan1_manual ] || fan=$(echo $smc/hwmon/hwmon*)
    echo "fancurve $(systemctl is-active fancurve), system-crash-capture $(systemctl is-active system-crash-capture)," \
        "fan $(cat $fan/fan1_input) rpm"
    echo "thunderbolt_net throttling=$(cat /sys/module/thunderbolt_net/parameters/throttling 2>/dev/null)"
} > "$R/system.txt" 2>&1
cat "$R/system.txt"
sudo -n dmesg > "$R/dmesg-full.txt" 2>&1
grep -E "WARNING|BUG|Oops|Call Trace|[Ee]rror|[Ff]ail|taint" "$R/dmesg-full.txt" > "$R/dmesg-warn.txt"
echo "dmesg: $(grep -c WARNING "$R/dmesg-warn.txt") WARNING, $(grep -c -E "BUG|Oops" "$R/dmesg-warn.txt") BUG/Oops lines"
{
    for p in /sys/bus/pci/drivers/amdgpu/0000:*; do
        h=$(echo $p/hwmon/hwmon*)
        clk() { echo "sclk $(($(cat $h/freq1_input) / 1000000)) mclk $(($(cat $h/freq2_input) / 1000000))"; }
        echo "$(basename $p) level $(cat $p/power_dpm_force_performance_level), $(clk)"
        echo high | sudo -n tee $p/power_dpm_force_performance_level > /dev/null; echo "  echo high: rc $?"
        sleep 2
        echo "  level $(cat $p/power_dpm_force_performance_level), $(clk)"
        echo auto | sudo -n tee $p/power_dpm_force_performance_level > /dev/null; echo "  echo auto: rc $?"
    done
} > "$R/dpm.txt" 2>&1
cat "$R/dpm.txt"
scripts/clockmon.sh "$R/clocks/xgpu.txt" & mon=$!
(cd ~/trashcan/bench/xgpu && ./xgpu p2p-verify) > "$R/xgpu-verify.txt" 2>&1; echo "exit $?" >> "$R/xgpu-verify.txt"
kill $mon; wait $mon 2>/dev/null
tail -2 "$R/xgpu-verify.txt"
scripts/clocksummary.sh "$R/clocks/xgpu.txt"
