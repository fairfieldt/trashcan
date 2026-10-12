#!/bin/bash
# Run a command with both D700s held at a fixed amdgpu DPM level (default high), restoring auto afterwards.
# usage: dpm-run.sh [LEVEL] -- CMD ARGS...
set -u
level=high
[ "${1:-}" != -- ] && { level=$1; shift; }
[ "${1:-}" = -- ] && shift
set_level() { for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do echo "$1" | sudo -n tee "$f" > /dev/null; done; }
trap 'set_level auto' EXIT INT TERM
set_level "$level"
echo "[dpm] $level: $(cat /sys/class/drm/card*/device/power_dpm_force_performance_level | tr '\n' ' ')" >&2
"$@"
