#!/bin/bash
# Samples both D700s' core/memory clocks (MHz) and temperature (C) every 0.25 s until killed.
# card0 is PCI 02:00.0, card1 is PCI 06:00.0. ggml's Vulkan0 is 06:00.0 (deviceUUID 00000000-0600-...).
# usage: clockmon.sh OUTFILE
out=$1
h0=$(echo /sys/class/drm/card0/device/hwmon/hwmon*)
h1=$(echo /sys/class/drm/card1/device/hwmon/hwmon*)
echo "t sclk0 mclk0 temp0 sclk1 mclk1 temp1" > "$out"
while :; do
    echo "$(date +%s.%N | cut -c1-14)" \
        $(( $(cat $h0/freq1_input) / 1000000 )) $(( $(cat $h0/freq2_input) / 1000000 )) $(( $(cat $h0/temp1_input) / 1000 )) \
        $(( $(cat $h1/freq1_input) / 1000000 )) $(( $(cat $h1/freq2_input) / 1000000 )) $(( $(cat $h1/temp1_input) / 1000 )) >> "$out"
    sleep 0.25
done
