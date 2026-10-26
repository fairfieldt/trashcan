# hwmon: (applesmc) Mac Pro 6,1 port lights

Two patches for the hwmon tree, from `kernel/patches-7.3/0009-applesmc-mac-pro-port-lights.patch` and
`0010-applesmc-port-lights-levels.patch`: the 20 LEDs behind the Mac Pro 6,1's I/O panel as LED class devices plus
the SMC's own light-show settings (1/2), and `port_lights_levels`, all LEDs in one write (2/2). The series also adds
the new ABI file to the APPLE SMC DRIVER entry in MAINTAINERS. Not sent.

**Status: ready for maintainer review.** It is a new feature, so send it after v7.4-rc1 (the 7.4 merge window is open
on 2026-10-26) and rerun the checks if hwmon-next has moved.

## Base

`git://git.kernel.org/pub/scm/linux/kernel/git/groeck/linux-staging.git` `hwmon-next` at
e1b3c2630d5b8bc755fe4ba86d20d08c734eb504 ("hwmon: (npcm750-pwm-fan) serialize PWM register updates"), fetched
2026-10-26. `drivers/hwmon/applesmc.c` there is the same as in v7.3 and in mainline d135ab453a74; the series applies to
both.

## Changes since the 2026-10-19 draft

- Rebased on hwmon-next (no conflicts; the C code is unchanged).
- `Assisted-by: LLM` removed (see `../README.md`); tool use is described in the cover letter.
- ABI `KernelVersion` 7.4 -> 7.5: 7.4's merge window is already open. Adjust `Date`/`KernelVersion` if it lands later.
- Cover letter: the measured cost per frame for 2/2, the six-Mac use and the untested retry path.

## Files

- `series/0000-cover-letter.patch`, `series/0001-...port-.patch`, `series/0002-...por.patch`: to send.
- `cover.txt`: cover letter source. `check.conf`: base and build settings for `../check-series.sh`.

## Checks (2026-10-26, kbuild VM)

`../check-series.sh applesmc-port-lights LOGDIR` (x86_64 defconfig + `INPUT`, `HWMON`, `SENSORS_APPLESMC=m`;
`x86_64-linux-gnu-gcc` 14.2.0; sparse v0.6.5-rc1-21-g84ec0658 from git, since Debian's 0.6.4 fails the kernel's
`checker-valid.sh`). Logs in `results/kernel/upstream-20261026/applesmc-port-lights/`:

- `make W=1 C=2 drivers/hwmon/applesmc.o` on the base, after 1/2 and after 2/2: no warnings from gcc or sparse
  (`build-main-*.log`, `new-warnings-main.txt` empty).
- `checkpatch.pl --strict`: only `MISSING_SIGN_OFF` on each patch (`checkpatch.log`).
- `get_maintainer.pl` (`get_maintainer-roles.txt`): Guenter Roeck (HARDWARE MONITORING), Henrik Rydberg (APPLE SMC
  DRIVER), linux-hwmon, linux-kernel. The "GitAuthor: Tom Fairfield" line comes from the series' own commits.

## Hardware tested

- indite (Mac Pro 6,1, SMC 2.20f18, LED controllers 1.4.a.0.6), 7.3.0-rc7-p2p-s2 with the code as a module override
  (`kernel/applesmc-override.sh`), 2026-10-18/19: LED map by eye, both animations, power light modes, triggers,
  bulk-write cost (8.05 ms for 20 changed LEDs, 2.15 ms for 5, 0.02 ms for none), bulk/single ordering race 30/30,
  motion switch-back race, unload/reload with fancurve running. Details: `results/kernel/port-lights-20261018/`.
- All six Mac Pros, 2026-10-26: the same module (md5 `506d510a641b...`, built from an `applesmc.c` identical to this
  series' result) loaded on each, 20 `smc::` LEDs registered; the lights worked on all six, including fast updates
  from the `portlights` music visualizer (40 frames/s through `port_lights_levels`) and activity lights during
  inference.

## Test limits

- A failed bulk write followed by a retry is code-reviewed only: there is no way to make an SMC write fail on demand,
  and no fault injection or mock was run. The cover letter says so.
- No head-to-head timing of 20 LED class writes against one `port_lights_levels` write; the justification is the
  per-transaction cost and the skipped unchanged LEDs, as measured.
- Only Mac Pro 6,1 with SMC 2.20f18. Other machines with the keys get generic `smc::port-light-<n>` names, untested.
- Built for x86_64 only (the driver depends on X86).

## Before sending

1. Read both patches and the cover letter; the "How this was made" and "Testing" paragraphs speak for you.
2. Sign off (in this directory): `perl -0pi -e 's/\n---\n/\nSigned-off-by: Tom Fairfield <tom\@fairfieldt.com>\n---\n/' series/000[1-9]*.patch`
3. In a kernel tree at the base: `./scripts/checkpatch.pl --strict PATH/series/000[1-9]*.patch` (expect no output
   other than "has no obvious style problems").
4. `git send-email --dry-run ...` with the command below, then without `--dry-run`.

```sh
git send-email --to="Guenter Roeck <linux@roeck-us.net>" --to=linux-hwmon@vger.kernel.org \
  --cc="Henrik Rydberg <rydberg@bitmath.org>" --cc=linux-kernel@vger.kernel.org series/*.patch
```
