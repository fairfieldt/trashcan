# Mac Pro 6,1 port lights (2026-10-18/19, indite)

The LEDs behind the I/O panel that light up when the Mac is moved are driven by the SMC ("Lightshow", three
controllers on the SMC's own I2C bus at 0x90/0x92/0x94, not visible to the host). Mainline applesmc has no support.
`kernel/patches-7.3/0009-applesmc-mac-pro-port-lights.patch` adds it.

## Using it

```bash
L=/sys/class/leds; D=/sys/devices/platform/applesmc.768
echo 255 > $L/smc::usb-box-top/brightness   # 0-255; switches the panel to manual mode
echo heartbeat > $L/smc::power/trigger       # any LED trigger works (modprobe ledtrig-heartbeat first)
echo auto > $D/port_lights                   # back to the SMC (lights on motion); "manual" reapplies the levels
echo 0 > $D/port_lights_motion               # no lighting on motion
echo 3000 > $D/port_lights_hold_ms           # time lit after motion (default 15000)
echo 1 > $D/port_lights_show                 # light up as on motion (automatic mode only)
```

In manual mode every LED shows its last set level (0 if never set). Unloading the module hands the lights back to the
SMC. The settings live in SMC RAM; a power loss restores the defaults.

| LED | Position | LED | Position |
|---|---|---|---|
| `audio-left` | left audio jack | `ethernet-box-left` | Ethernet outline, left |
| `audio-right` | right audio jack | `usb-box-top` | USB outline, top |
| `usb` | USB, center | `usb-box-bottom` | USB outline, bottom |
| `thunderbolt` | Thunderbolt, center | `thunderbolt-box-top` | Thunderbolt outline, top |
| `ethernet` | Ethernet, center | `thunderbolt-box-bottom` | Thunderbolt outline, bottom |
| `hdmi` | HDMI | `ethernet-box-top` | Ethernet outline, top |
| `usb-box-right` | USB outline, right | `ethernet-box-bottom` | Ethernet outline, bottom |
| `usb-box-left` | USB outline, left | `power` | power light |
| `thunderbolt-box-right` | Thunderbolt outline, right | `power-2` | power light (brighter) |
| `thunderbolt-box-left` | Thunderbolt outline, left | | |
| `ethernet-box-right` | Ethernet outline, right | | |

The order is the SMC's LED index 0-19 (Tom watched each one light in turn).

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

## Tested on indite (7.3.0-rc7-p2p-s2)

Built as a module against the s2 headers in the kbuild VM and installed as
`/lib/modules/7.3.0-rc7-p2p-s2/updates/applesmc.ko` (overrides the stock module; taints the kernel O and E). It loads at
boot (`ports=20`), registers the 20 LEDs, and `fancurve` works with it. Checked: setting one LED switches to manual and
`port_lights` reads `manual`; `auto` returns to the SMC; `port_lights_hold_ms` 3000 and `port_lights_show` light the
panel for 3 s; `port_lights_motion` 0/1; bad input gives `EINVAL`; heartbeat and timer triggers on `smc::power` and
`smc::hdmi` (Tom watched); `rmmod` leaves `LIT!` 0 and `LsHS` 03.

The headers packages' host tools are mixed (objtool arm64, fixdep x86-64), so modules don't build on the Macs. To
build the override module in the kbuild VM: extract `linux-headers-7.3.0-rc7-p2p-s2_*.deb`, replace each ELF in its
`scripts/` and `tools/` with the arm64 one from `~/kernel/7.3/linux-7.3-rc7`, then
`make -C HEADERS M=DIR ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu- modules` with the patched `applesmc.c` and
`obj-m := applesmc.o` in `DIR`.

Once (during a test loop) `LIT!` was found 0 without anything writing it. The Lightshow init (0x172fa, from the
controller enumeration at 0x18b28) clears it; a 60 s hold test kept it. The driver reapplies manual mode on the next
brightness change.
