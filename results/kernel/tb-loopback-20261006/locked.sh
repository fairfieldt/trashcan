#!/bin/bash
# Run a benchmark command under the shared incite lock (~/trashcan-exp/.lock, also used by the engine
# run scripts) with the performance governor and turbo off; restore schedutil and turbo afterwards.
# usage: locked.sh CMD...
exec 9>"$HOME/trashcan-exp/.lock"
flock 9
restore() {
  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo schedutil | sudo tee $c >/dev/null; done
  echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
}
trap restore EXIT
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee $c >/dev/null; done
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
"$@"
