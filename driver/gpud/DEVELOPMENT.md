# Driver and desktop iteration on dross

`gpud-dev` is an opt-in developer supervisor, not a production application
launcher. Init gives it one PCI connection, the firmware framebuffer and program
images. Every administrative request requires a fresh 256-bit bearer token.
It is stored in the real mode-0600 regular file `/run/gpud/dev.token`; the
namespace-only Unix socket itself has no filesystem inode on dross. The client
reads that protected file and neither prints the secret nor puts it in argv.
Do not provision this service in an untrusted application environment. Its
administrative operations are separate from gpud's public rendering socket.

After one boot with this service, upload small executable builds over SSH to new
paths under `/tmp`, then use `gpud-devctl`. Driver replacement does not require
Linux or a machine reboot while the old epoch remains healthy.

    gpud-devctl check-auth
    gpud-devctl status
    gpud-devctl load windowd /tmp/oscar-windowd-next
    gpud-devctl run
    gpud-devctl load driver /tmp/gpud-next
    gpud-devctl restart
    gpud-devctl run
    gpud-devctl play
    gpud-devctl stop

`load` accepts the roles `driver`, `windowd`, `checker`, `doom` and `vkdraw`.
`session` is an optional native session image for `play`.
It bounds images to 8 MiB, checks the x86-64 ELF header and copies the image into
supervisor-owned native Memory before changing the selected image. The upload
file may then be removed. The currently running process keeps its old image.
`set KEY VALUE` updates the next launch's LD_LIBRARY_PATH, LD_PRELOAD,
VK_DRIVER_FILES, GPUD_SOCKET or IWAD (`iwad`) value. Stage shared libraries under
a new path before changing the preload value.

`run` launches the bounded native-window/scanout checker with the selected Doom,
Vulkan application and GPU windowd images. A successful run requires game/input
checks, native GPU surface checks, process close and firmware-restore
acknowledgement. It holds both windows visibly for ten seconds for physical
capture. A used display channel is never recycled: the next run first creates a
new driver epoch with fresh channels.

`play` starts native `oscar-session` with the real USB seat, GPU windowd and Doom.
It returns immediately and never synthesizes input or closes the game on a
timer. Close Doom’s window to end the session and restore firmware scanout. The
supervisor tracks the session and rejects restart or another launch while it is
alive. Init must also supply `@module/oscar-session=module/oscar-session` and the
physical seat channel as `@seat=input`; usbd receives the other end. Old queued
seat reports are drained before launch.

A restart sends a private native shutdown request. gpud refuses it while a client
is still connected, scanout is acquired, or a submitted fence is unfinished. Once
all are drained it stops the engines and exits successfully. The supervisor waits
for actual process cleanup, then asks pcid to release that owned function. pcid
checks the same owner and an exited Process handle, disables and reads back bus
mastering/MSIs, and only then drops its retained DMA domain. The new process gets
a fresh domain and its own PCI/bootstrap/display channels. Merely retaining a PCI
channel across `exit()` would retain stale mappings and collide with the new
process's IOVA allocator.

Every administrative wait is bounded: request input 2 s, driver ready 15 s,
shutdown acknowledgement 2 s, driver exit 3 s, pcid release 2 s, checker exit 60 s.
The interactive `play` session has no automatic close deadline.
A timeout or failed shutdown keeps the epoch and backing quarantined. It never
force-kills a driver or silently frees live GPU memory. An unrecoverable epoch
requires an explicitly coordinated reset/detach/reboot. Close clock monitors and
other GPU clients before restart. `status` reports retained driver/app handles;
use kernel/gpud logs for device state.

## Init provisioning

Include modules `gpud-dev`, `gpud`, `oscar-windowd-gpu`, `oscar-scanoutcheck`,
`oscar-doom`, `oscar-vkdraw`, and the pcid build with RELEASE support. Put
`gpud-devctl` in the Linux rootfs. Replace the ordinary gpud start and desktop
check with the following, keeping the existing network, USB and disk services:

    start gpud-dev @display=framebuffer @module/gpud=module/gpud @module/oscar-windowd-gpu=module/oscar-windowd-gpu @module/oscar-scanoutcheck=module/oscar-scanoutcheck @module/oscar-doom=module/oscar-doom @module/oscar-vkdraw=module/oscar-vkdraw %width=$display-width %height=$display-height %stride=$display-stride %format=$display-format %offset=0 %gpu-bus=6 %iwad=/disk/.oscar-hardware-20261103/freedoom1.wad %LD_LIBRARY_PATH=/disk/alpine-3.22/lib:/disk/alpine-3.22/usr/lib %LD_PRELOAD=/usr/lib/libgpud-new.so %VK_DRIVER_FILES=/disk/alpine-3.22/radeon_icd.json %GPUD_SOCKET=/run/gpud/gpu1.sock

Init supplies the supervisor's PCI connection automatically. Do not start a
second gpud for the same function. The supervisor creates its own private display
channel; there is no `channel scanout` binding in this configuration. The current
developer target is bus 6, socket gpu1, with the existing firmware mode retained.

## Hardware validation

On Indite on 2026-11-03, missing/wrong tokens were rejected and a valid token was
accepted. A connected clock monitor correctly prevented shutdown; after it
closed, upload/load/restart completed in 1.887 s without rebooting dross. Sixteen
sequential native Memory import clients each passed 32 exact frames and reused
the same page-directory address. Initial and post-replacement Doom/Vulkan tests
passed and acknowledged firmware restore.

The final owned-alias scene completed 840 flips. The independent import proof
revoked the caller's original mapping/handle before 32 exact GPU frames and
reported 4096-byte alignment. The physical `play` session was then left running
with the USB seat, without synthetic input or automatic close.

Evidence is in `results/dross/hardware-20261103`: `v6-auth-check.txt`,
`v6-active-client-guard-clocks.txt`, `v6-hot-restart.txt`, `v6-vm-reuse.txt`,
`v6-retire-16.txt`, `final-v8-alias-probe.txt`, `final-v8-acceptance.txt`,
`final-v8-netlog.txt`, `final-v8-play.txt` and `oscar-doom-indite.png`.
No intentional GPU hang/reset recovery was tested.
