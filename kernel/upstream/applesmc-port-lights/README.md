# Mac Pro 6,1 port lights for upstream

Upstream form of `kernel/patches-7.3/0009-applesmc-mac-pro-port-lights.patch` and
`0010-applesmc-port-lights-levels.patch`, as a two-patch series for the hwmon tree. The code is the same as the
local patches; the series also adds the ABI file to applesmc's MAINTAINERS entry.

## Status

- `series/`: `git format-patch --cover-letter` on v7.3 (`drivers/hwmon/applesmc.c` is the same in v7.3, master and
  hwmon-next as of 2026-10-19). Not sent.
- The patches carry `Assisted-by: LLM` (the format in v7.3's Documentation/process/coding-assistants.rst) and no
  `Signed-off-by`: only the submitter can certify the DCO.
- Recipients (`scripts/get_maintainer.pl`): To Guenter Roeck <linux@roeck-us.net> (hwmon) and
  linux-hwmon@vger.kernel.org; Cc Henrik Rydberg <rydberg@bitmath.org> (applesmc) and linux-kernel@vger.kernel.org.

Before sending:

1. Read both patches and the cover letter. The cover letter's "How this was made" paragraph speaks for you: check it
   says what you want to say.
2. Add your sign-off below `Assisted-by` in 0001 and 0002 (`Signed-off-by: Tom Fairfield <tom@fairfieldt.com>`), or
   `git am` the series and `git rebase --signoff`.
3. Rebase on hwmon-next if applesmc.c has changed there, and rerun `scripts/checkpatch.pl --strict`.
4. `git send-email --to linux@roeck-us.net --to linux-hwmon@vger.kernel.org --cc rydberg@bitmath.org
   --cc linux-kernel@vger.kernel.org series/*.patch`

## Evidence (2026-10-19)

- **checkpatch:** `--strict --no-signoff` reports nothing for either patch (with the sign-off check, only the missing
  sign-off).
- **Build:** each stage builds with W=1 and no warnings (against the 7.3.0-rc7-p2p-s2 headers in the kbuild VM).
- **Hardware:** indite (Mac Pro 6,1, SMC 2.20f18), running the same code as a module override
  (`kernel/applesmc-override.sh`): the LED map (Tom, by eye), both animations and the power light modes (Tom),
  bulk-write speed (8.05 ms for 20 changed LEDs, 2.15 ms for 5), the bulk/single ordering race (30/30) and the motion
  race (a write during the Hello switch-back waited 195 ms and stuck), unload and reload with fancurve running.
  Details in `results/kernel/port-lights-20261018/`.
- Not tested: repeating a failed bulk write (no way to make an SMC write fail on demand); sparse (not installed).
