# Upstream packages refreshed (A18, 2026-10-26)

Logs of `kernel/upstream/check-series.sh` and the test fixtures for the packages in `kernel/upstream/` (see its
README for the queue and status). Nothing was sent, signed off or installed. No GPU was used and no Mac was changed.

Environment: kbuild Lima VM (Debian 13, arm64), `x86_64-linux-gnu-gcc` 14.2.0, x86_64 defconfig plus each package's
options, sparse v0.6.5-rc1-21-g84ec0658 built from git.kernel.org (Debian's 0.6.4 fails the kernel's
`checker-valid.sh`, so `C=2` would silently skip it). Kernel repository `~/a18/linux` in the VM with remotes torvalds,
hwmon (groeck/linux-staging), pci, tb (westeri/thunderbolt), next and amd (gitlab.freedesktop.org/agd5f/linux),
fetched 2026-10-26:

| Ref | Commit |
|---|---|
| torvalds/master | d135ab453a74 (7.4 merge window) |
| hwmon/hwmon-next | e1b3c2630d5b |
| pci/next | 17bb0e2f6c80 |
| tb/next, tb/fixes | a93a8e320000, 96a8479804e3 |
| amd/amd-staging-drm-next | 77f548b127c6 |
| next/master | f1ac44c32cba (next-20261026) |

Per package directory:

- `check.log`: the run (base, apply checks on other refs, configs, per-stage build results, checkpatch summary,
  get_maintainer). `build-CONFIG-STAGE.log`: W=1 + sparse output for the touched objects on the base and after each
  patch; `new-warnings-CONFIG.txt`: lines the series adds (all empty).
- `checkpatch.log`: `checkpatch.pl --strict --show-types` on the patches (cover letter excluded). Only
  `MISSING_SIGN_OFF` everywhere, plus 8 `CAMELCASE` checks on the amdgpu VFCT patch (atombios field names).
- `get_maintainer.txt`, `get_maintainer-roles.txt`.
- `thunderbolt-stream-hopid/hopid-replay.txt`: `kernel/upstream/thunderbolt-stream-hopid/harness/run.sh` on `stream.c`
  before and after the patch (5 of 6 scenarios free a HopID the stream doesn't hold before, none after).
- `vfct-exact-match/vfct-replay.txt`: `kernel/upstream/vfct-exact-match/harness/run.sh` on the rebased tree against
  the base with incite's VFCT (dump not tracked: it contains the VBIOS images).

`macs-readonly.txt`: read-only checks over SSH on all six Macs: kernel, port-light LED count, md5 of the applesmc
override module (the same on all six, built from an `applesmc.c` identical to the upstream series' result), and each
GPU's `vbios_version` (each D700 has its own VBIOS under the amdgpu VFCT patch).

Rerun one package (in the VM, with the remotes above fetched):
`kernel/upstream/check-series.sh kernel/upstream/PACKAGE LOGDIR`. It refuses to run if the target branch has moved
from `BASE_COMMIT` in the package's `check.conf`.
