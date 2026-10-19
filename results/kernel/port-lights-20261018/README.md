# Mac Pro 6,1 port lights (2026-10-18/19, indite)

The LEDs behind the I/O panel that light up when the Mac is moved are driven by the SMC ("Lightshow", three
controllers on the SMC's own I2C bus at 0x90/0x92/0x94, not visible to the host). Mainline applesmc has no support.
`kernel/patches-7.3/0009-applesmc-mac-pro-port-lights.patch` adds it, with ABI documentation
(`Documentation/ABI/testing/sysfs-devices-platform-applesmc`); `0010-applesmc-port-lights-levels.patch` adds
`port_lights_levels` on top.

## Using it

```bash
L=/sys/class/leds; D=/sys/devices/platform/applesmc.768
echo 255 > $L/smc::usb-box-top/brightness   # 0-255; switches the panel to manual mode
echo heartbeat > $L/smc::power/trigger       # any LED trigger works (modprobe ledtrig-heartbeat first)
echo auto > $D/port_lights                   # back to the SMC (lights on motion); "manual" reapplies the levels
echo 0 > $D/port_lights_motion               # no lighting on motion
echo 3000 > $D/port_lights_hold_ms           # time lit after motion (default 15000)
echo 1 > $D/port_lights_show                 # light up as on motion (automatic mode only)
echo you-again > $D/port_lights_show         # the other animation (hello: the motion one)
echo breathe > $D/power_light                # off, on, breathe, bright (while automatic)
echo 0 0 255 0 ... > $D/port_lights_levels   # all 20 at once (0-255, SMC order); only changed LEDs go to the SMC
```

Each changed LED costs one SMC transaction, about 0.4 ms: a frame where all 20 change takes 8.1 ms (123 fps at
most), one where 5 change 2.2 ms, and an unchanged frame nothing. The driver rechecks that the SMC is still in manual
mode at most once a second (reading `LIT!` costs another transaction). `portlights` in the homelab repo drives the
lights from audio and GPU activity.

In manual mode every LED shows its last set level (0 if never set). Unloading the module hands the lights back to the
SMC. The settings live in SMC RAM; a power loss restores the defaults.

| LED | Position | LED | Position |
|---|---|---|---|
| `line-out` | audio line out (left jack) | `ethernet-box-left` | Ethernet outline, left |
| `headphone` | headphone jack (right) | `usb-box-top` | USB outline, top |
| `usb` | USB, center | `usb-box-bottom` | USB outline, bottom |
| `thunderbolt` | Thunderbolt, center | `thunderbolt-box-top` | Thunderbolt outline, top |
| `ethernet` | Ethernet, center | `thunderbolt-box-bottom` | Thunderbolt outline, bottom |
| `hdmi` | HDMI | `ethernet-box-top` | Ethernet outline, top |
| `usb-box-right` | USB outline, right | `ethernet-box-bottom` | Ethernet outline, bottom |
| `usb-box-left` | USB outline, left | `power` | power light |
| `thunderbolt-box-right` | Thunderbolt outline, right | `power-2` | power light (brighter) |
| `thunderbolt-box-left` | Thunderbolt outline, left | | |
| `ethernet-box-right` | Ethernet outline, right | | |

The order is the SMC's LED index 0-19 (Tom watched each one light in turn). The names are ours: the firmware only
numbers the LEDs, and Apple's documentation names the ports (audio line out, headphone, enclosure latch, USB 3,
Thunderbolt 2, Gigabit Ethernet, HDMI) but not the lights. The enclosure latch, left of the line out jack, has no
light.

## Animations and the power light

The firmware has two animations, named in its debug output (`LIGO=10` prints both per LED: up delay, up rate, down
delay, down rate, brightness):

- **Hello** (`LIGO=8`): the lights sweep on from the bottom up and off from the top down. Motion shows this one.
- **You Again** (`LIGO=9`): all at once.

Selecting one lights the panel too, and the choice stays: nothing in the SMC switches back, so motion would show
You Again from then on. After `you-again` the driver switches back to Hello once the lights have faded (the hold
time and 3 s), with `LIEN` off for 200 ms so the SMC drops the light-up that comes with it.

`LSSB` (2 bytes: mode, 0) sets the power light's mode, with the firmware's names `LmsOff`, `LmsOn`, `LmsBreathe` and
`LmsBrightNoScale`: 0 off, 1 on at `LSLN` (0xaaaa), 2 breathing (the sleeping Mac's slow pulse), 3 full (0xffff
whatever `LSLN` says). It applies while the port lights are automatic (the firmware copies the power light into LEDs
18 and 19). `LSSS` takes the same arguments and also stores the mode.

## SMC keys

From the SMC firmware 2.20f18 (Apple's `MacProSMCUpdate.dmg`,
https://updates.cdn-apple.com/2019/cert/041-88228-20191011-7e93f96c-b0c1-41ad-8e39-ebe6b37d3d5d/MacProSMCUpdate.dmg,
sha256 `26bd314852215590bf563e4837f3d8a3d41eb896f5b480cb89ab4521323c4c67`; the image is
`2013MACPRO61.smc` in its payload). `smc2bin.py` turns it into a flat Cortex-M image; the key table is at 0x2f000
(16-byte entries: name, flags, length, 2 pad, type, pointer to a RAM variable or a handler). `fwdis.py` disassembles
with capstone and names key variables. The Lightshow code is at 0x17000-0x19700, its state a struct at 0x20000a00.

| Key | Type | Meaning |
|---|---|---|
| `LIGO` | ui8, write-only | Command. 1: light up (the accelerometer's motion handler at 0x203d8 calls the same code, 0x179e6). 3/4: gamma on/off; 8/9: animation profile A/B; 10: print the config to the SMC log |
| `LIEN` | flag | Motion and `LIGO` lighting enabled |
| `LIHD` | ui16 | Time lit, 10 ms units (1500) |
| `LIT!` | flag | Manual mode: LEDs show the `LInD` levels instead of the animation |
| `LInD` | be16 level, ui8 index | Manual level of one LED, applied at once; index < `LINm` (20) |
| `LITC` | flag | Per-LED calibration scaling (1) |
| `LInS`..`LInW` | be16, index, profile | Per-LED animation profile parameters |
| `LIFU`/`LIFD`/`LIFA`/`LIFC`/`LIYA` | ui16 | Fade parameters |
| `LSSB` | mode, 0 | Power light: off, on, breathe, bright |
| `LsbR` | 3 bytes | Readback of one channel: 0, then little-endian inverted PWM (`ff ff` off, `02 00` full) |
| `LSLN` | {pwm | Power light level in automatic mode; the firmware mirrors it to LEDs 18 and 19 |
| `Lsb*`, `Lsm*`, `LsEB`, `LsEF` | | Controller firmware update and factory data. Don't write |

Writing `LsEB=01` (by mistake, while probing) switched the Lightshow handler to `LsHS=04`: all three controllers read
back zeros, and a reboot didn't help. Shutting down and unplugging the Mac for 15 s restored it (firmware intact,
`LsbV` 1.4.a.0.6).

## Files

- `smc.py`: SMC access over `/dev/port` for exploring (`systemctl stop fancurve; rmmod applesmc` first, as it
  bypasses the driver). `list`, `info`, `read`, `write`, `watch` (print key changes), `probe` (write, then watch `LsbR`).
- `indite-L-keys.txt`: every `L*` key with its flags (0x80 read, 0x40 write, 0x10 function) and value (serial redacted).
- `indite-nudge-watch.txt`: the keys that changed when Tom nudged indite (accelerometer readings removed).

## Review (2026-10-19)

A review on hardline (codex-review-fd40, "applesmc Mac Pro 6,1 port lights: correctness review and bulk-write API
assessment") found three races, fixed:

- A failed bulk write couldn't be retried: the levels were recorded before they were sent. Each LED now has an
  "unsent" bit, cleared only when its SMC write succeeds; a repeated write sends what didn't get through. Not tested on
  hardware (no way to make an SMC write fail).
- A single-LED change still queued in the LED core could land after a later bulk write. The bulk write now flushes
  each LED's pending work first. On indite, 30 rounds of `brightness=255` then a bulk write of 64 ended at 64 every
  time (`LsbR`, which reads LED 12, `usb-box-top`).
- `port_lights_motion` (and the other setting writes) didn't take the lock, so the Hello switch-back, which turns
  motion off for 200 ms and restores what it found, could undo a write made meanwhile. On indite a write 50 ms into the
  switch waited 195 ms and stuck.

As it suggested, the bulk interface is a patch of its own (0010) with a stated contract (ABI file): equivalent to
writing each LED in order, LEDs with a trigger keep it, reads give the LED class brightness, a failed write can be
repeated. All new files are documented.

## Tested on indite (7.3.0-rc7-p2p-s2)

Built as a module against the s2 headers in the kbuild VM and installed as
`/lib/modules/7.3.0-rc7-p2p-s2/updates/applesmc.ko` (overrides the stock module; taints the kernel O and E). It loads at
boot (`ports=20`), registers the 20 LEDs, and `fancurve` works with it. Checked: setting one LED switches to manual and
`port_lights` reads `manual`; `auto` returns to the SMC; `port_lights_hold_ms` 3000 and `port_lights_show` light the
panel for 3 s; `port_lights_motion` 0/1; bad input gives `EINVAL`; heartbeat and timer triggers on `smc::power` and
`smc::hdmi` (Tom watched); `rmmod` leaves `LIT!` 0 and `LsHS` 03.

The headers packages' host tools are mixed (objtool arm64, fixdep x86-64), so modules don't build on the Macs.
`kernel/applesmc-override.sh HOST...` builds the override in the kbuild VM against the host's installed headers
(with the VM tree's arm64 host tools swapped in), installs it and reloads applesmc.

Once (during a test loop) `LIT!` was found 0 without anything writing it. The Lightshow init (0x172fa, from the
controller enumeration at 0x18b28) clears it; a 60 s hold test kept it. The driver reapplies manual mode on the next
brightness change.
