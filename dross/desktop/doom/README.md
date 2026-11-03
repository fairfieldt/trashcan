# Doom in an Oscar window

DoomGeneric uses Oscar's native channels and server-created shared XRGB8888
surfaces. The musl executable runs under dross's Linux compatibility layer, with
no SDL, X11 or Wayland dependency. The classic game rasterizer runs on the CPU;
windowd composes its surface using either renderer. Audio and mouse look are not
implemented. Losing focus releases held keys.

## Controls

Click the window to focus it. Hold Control/Ctrl to fire; Command is not mapped
to fire. Mouse clicks focus or move the window rather than aiming or firing.

| Key | Action |
|---|---|
| Up / Down | Move forward / backward |
| Left / Right | Turn |
| Ctrl | Fire |
| Space | Open doors and use switches |
| Shift | Run |
| Alt/Option + Left / Right | Strafe |
| 1–7 | Select an owned weapon |
| Tab | Toggle map |
| Esc | Open menu and pause |
| Minus / Equals | Decrease / increase view size |

Menus use arrows, Enter to select, Backspace to go back, and Y/N for confirmations.
F1 opens help; F2/F3 save/load; F4 opens volume; F5 changes detail; F6/F9 quicksave/
quickload; F7 ends the game; F8 toggles messages; F10 quits. Audio is not implemented.
On the map, Minus/Equals zoom, 0 toggles full-map zoom, F toggles follow, G toggles
the grid, M adds a marker and C clears markers. F11/F12, dedicated strafe keys and
mouse-look input are not currently forwarded by the adapter.

## Build

Build:

    ./build-musl.sh

This creates `../target/x86_64-unknown-linux-musl/release/oscar-doom` and fetches
Freedoom 0.13.0 into ignored `../target/freedoom/`. The archive is checked against
the upstream SHA-256. Copyright/redistribution notices are kept alongside the WAD.
The DoomGeneric revision and engine license are in PROVENANCE.md. No proprietary
Doom IWAD is shipped.

Start the executable with a bootstrap `@window` channel connected to windowd, and
arguments `-iwad PATH/freedoom1.wad -warp 1 1 -nosound`. The WAD filesystem may be
read-only. The current launcher uses the RAM working directory for saves/config;
those files do not survive an OS reboot. GPU windowd uses Mesa through Linux
compatibility without changing the native window API, which remains provisional.

## Acceptance

Run the isolated QEMU image:

    ./test-qemu.py
    SMP=1 ./test-qemu.py

It builds a separate ESP under `../target/doom-qemu/`, leaving the normal ESP,
physical disks and network boot configuration untouched. The test starts real
Doom and native windowd on dross, waits for a playable level, checks advancing
frames and player movement from held input, creates an occluding native window,
checks focused input and shared-buffer acknowledgements, drags/closes the game,
and waits for both processes to exit. `qemu.log` records results and
`oscar-doom.ppm` captures actual QEMU firmware scanout. One-CPU and four-CPU runs
passed on 2026-11-03 using the CPU compositor.

The test-only `acceptance` feature exposes game-state observations solely through
a supplied test channel; normal builds have no observer. Build it explicitly for
hardware checks:

    DOOM_FEATURES=acceptance ./build-musl.sh

The physical Indite Mac Pro then passed the same native-window checks with GPU
windowd, Mesa RADV and gpud using firmware framebuffer copying. The separate
`oscar-scanoutcheck` runner tests private current-mode page flips, actual
held/released key-state packets and a native Vulkan app beside Doom. It keeps Doom
firing for ten seconds for physical capture, closes both apps, and requires GPU
windowd to acknowledge firmware restoration. It does not read the old firmware
framebuffer to pretend to verify a page-flipped surface.

The final owned-alias cohort passed on the 12-CPU Mac Pro: 840 completed flips on
CRTC3, counter 5786 to 6682, 35 game tics per second during the firing interval,
native GPU window pixels/pointer input/FrameDone checks, and successful firmware
restore. Evidence is in `results/dross/hardware-20261103`:
`final-v8-netlog.txt`, `final-v8-acceptance.txt`, `final-v8-manifest.json` and
`final-v8-acceptance-camera/006.png`. These are correctness observations, not a
sustained frame-rate comparison. See [the hardware recipe](../gpu/HARDWARE.md).

After validation, `gpud-devctl play` launches native `oscar-session` with the real
USB keyboard/mouse seat. It does not synthesize movement or close Doom on a timer.
The final session was left visible for physical play; `final-v8-play.txt` records
the launch and `oscar-doom-indite.png` captures its real level. Close the Doom
window to restore firmware presentation and allow another driver replacement.
