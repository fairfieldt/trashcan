# Times reads of the SMC fan speed (what fancurve reads each interval): N reads, median/p95/max in ms.
import glob, sys, time, statistics
f = glob.glob('/sys/devices/platform/applesmc.768/hwmon/hwmon*/fan1_input')[0]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
t = []
for _ in range(n):
    a = time.perf_counter()
    with open(f) as h:
        h.read()
    t.append((time.perf_counter() - a) * 1e3)
    time.sleep(0.01)
t.sort()
print(f"{sys.argv[2] if len(sys.argv) > 2 else ''}: n={n} median {statistics.median(t):.2f} ms, p95 {t[int(.95*n)]:.2f}, max {t[-1]:.2f}, mean {statistics.mean(t):.2f}")
