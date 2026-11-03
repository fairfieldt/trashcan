# Vulkan rendering on Oscar

`oscar-windowd-gpu` is a musl executable. It implements the same native OSCR
channels and shared-window extension as native windowd, using Vulkan graphics
through Mesa RADV and gpud. Linux compatibility is an adapter dependency; clients
need neither Linux file descriptors nor Mesa types in the window protocol.

The renderer uses actual graphics pipelines: triangles, fragment sampling of
opaque window pixels, alpha-blended glyphs, clipped fills and a stippled desktop.
CPU and GPU rendering share window policy through `canvas::Painter`.

Build with `./build-musl.sh`. This uses dynamic musl because Mesa libraries are
loaded with `dlopen`. `oscar-vkcheck` tests offscreen graphics pixels; `oscar-vkdraw`
is a small GPU-drawn application using the native shared-window API. It imports
that window’s native Memory as the Vulkan transfer destination; it does not copy
the frame through a CPU pixel buffer. `oscar-vkshare` checks 32 exact frames in an
imported native Memory object before that path is used for presentation.

A session supervisor selects the GPU server when explicitly supplied
`@module/oscar-windowd-gpu`; it otherwise uses native `@module/oscar-windowd`.
It forwards bootstrap values `LD_LIBRARY_PATH`, `LD_PRELOAD`, `VK_DRIVER_FILES`
and `GPUD_SOCKET` as the GPU server's environment, plus `gpu-bus` as a device
selection value. The normal gpud/RADV setup is documented in `../../README.md`.
Both adapters receive the same exclusive display and trusted input capabilities.

## Validation and display handover

Host correctness:

    cargo run -p oscar-desktop-gpu --bin oscar-vkcheck -- --library /opt/homebrew/lib/libMoltenVK.dylib

On a Linux D700 with the system Vulkan loader:

    oscar-vkcheck --bus 2

For gpud, run the same binary with the existing gpud DRM preload and socket
configuration. GPU runs on shared lab machines require their gpuq lease.

On 2026-11-03, exact solid, clipped texture and desktop-pattern pixels passed on
MoltenVK/Apple M3 Ultra and RADV TAHITI/D700 under Linux amdgpu. Full Oscar scenes
(overlapping windows, focus, dragging, text and disconnect) matched the software
renderer, allowing at most one code value of glyph blend rounding per channel.
Hardware source/commands/clocks are in `results/driver/oscar-vulkan-20261103`.
These are correctness results, not a desktop speedup measurement.

On 2026-11-03 the same offscreen checks also passed through gpud and Mesa RADV on
Indite running dross. The dynamic musl executable was required: linking Zig’s
bundled static libc made `dlopen` fail before reaching Vulkan. Real-hardware
logs are in `results/dross/hardware-20261103/vkcheck-existing-dross-dynamic.txt`.

Current-mode display handover is implemented in `driver/shim/core/display.c`,
using the retained Linux DCE6 register definitions and page-flip sequence. It
captures active firmware planes before VRAM allocation, reserves their backing
in both allocators, and tells SI power management that the displays remain live.
It keeps firmware connector training, timings, format and clock state. This is
not a connector/mode-setting API.

Only the supervisor-supplied private native display channel can acquire scanout;
the public gpud render socket has no display operations. Windowd imports the
server’s staging Memory into Vulkan. A completed graphics fence precedes the
SDMA copy into alternating VRAM surfaces. Each flip clears the old page-flip
status, then must observe a fresh DCE6 page-flip event, the requested primary address, a cleared update-pending bit
and an advancing CRTC frame counter before success. This observes vblank-latched completion; it is not a kernel
interrupt subscription. Owner close restores the firmware plane before freeing surfaces.
The original framebuffer stays reserved throughout. A kernel panic still draws
into that original framebuffer; while another plane is scanned out, serial or
network logs are needed until firmware ownership is restored or the machine
reboots. Kernel-panic display restoration is not implemented.

On Indite’s 12-CPU dross boot, the updated gpud/preload passed 32 exact native
Memory import frames. GPU windowd also ran real Doom in a native Oscar window:
changing displayed pixels, held forward movement, focused input, a second
occluding window, dragging and clean process exit all passed. Physical camera
frames `results/dross/hardware-20261103/gpu2-camera/036.png` and `037.png` show
the playable level and firing pistol. `gpu2-netlog.txt`, `gpu2-manifest.json` and
`gpu-check2.conf` in the same archive record checks, binaries, configuration and
4 Hz clocks. During the short Doom run the display D700 reported 850 MHz core,
1370 MHz memory and 45–46°C. This is correctness evidence, not sustained workload
throughput or a CPU-versus-GPU comparison. Presentation in that test copied to
the firmware framebuffer.

The subsequent physical scanout test also passed on Indite. Doom ran beside a
native Vulkan application for a ten-second firing capture. Game state advanced
35 tics per second; the Vulkan app checked pixels written directly into its
native Window Memory, received pointer input and matched FrameDone replies.
Windowd completed 830 vblank-latched flips on CRTC3 (frame counter 4372 to 5263),
then both applications closed and firmware restoration was acknowledged by both
windowd and gpud. Evidence is `results/dross/hardware-20261103/v6-auth-netlog.txt`,
`v6-auth-run1.txt` and `v6-auth-boot-camera/069.png`. Tahiti's INUSE register aliases
read zero; completion uses Linux DCE6's page-flip status rather than those aliases.
The final owned-alias cohort repeated the whole physical test successfully:
840 completed flips, CRTC3 frame counter 5786 to 6682, all native-window/Doom/GPU
application checks passed, and firmware restore returned 0 with windowd's ACK.
The ten-second firing interval advanced exactly 35 game tics per second. The
capture is `results/dross/hardware-20261103/final-v8-acceptance-camera/006.png`;
`final-v8-netlog.txt`, `final-v8-acceptance.txt` and `final-v8-manifest.json` record
checks, binary hashes and 4 Hz clocks (850 MHz core, 1370 MHz memory, 47–48°C
while active). `final-v8-alias-probe.txt` records the separate 32-frame proof
with the original mapping/handle revoked and 4096-byte import alignment.

These counts and game tics are correctness observations, not a general frame-rate
benchmark. GPU desktop/inference contention and sustained end-to-end performance
remain unmeasured. These checks do not establish a speedup or general recovery.

## Lifetime and bounded waits

Rendering recording is bounded to 3 Mi words and 8192 draws per frame. A rendering
submission has a 500 ms fence deadline and never waits on a client-owned GPU
event. A failed fence poisons the renderer; it retains Vulkan objects and their
dynamic library until process exit.

The updated gpud retirement path fixes the earlier unsafe `bo_free` behavior:
a 500 ms timeout returns an error without freeing the BO, recycling its file
range, or dropping its IOMMU mapping. Disconnected clients retain all backing and
their VMID until every used ring’s last fence is observed complete. Imported
Memory has an independently held fd, CPU mapping and DMA mapping. The renderer
also owns a separate process mapping of that Memory. A poisoned renderer keeps
this alias and Vulkan’s dynamic library alive until process exit, so dropping
the caller’s Window or Scanout cannot invalidate a worker thread’s host pointer. Shutdown
refuses to discard unfinished work or an acquired display. Display-copy timeout
retains staging until copy completion; failed firmware restore keeps the display
allocation quarantined. An unrecoverable device requires an operator-controlled
reset/detach/reboot before backing may be forcibly released. These failure paths
have unit coverage for fence ordering/wraparound; intentional GPU hangs have not
been injected on hardware.

## Private display protocol (provisional)

Init creates a channel, passes one end to gpud as bootstrap `display` and the
other to the session as `scanout`. It does not give this channel to applications.
Requests are two little-endian u32s: operation and argument. Operation 0 acquires
a CRTC (argument `u32::MAX` chooses an active supported plane); its reply is i64
status, eight u32s (CRTC, width, height, pitch in pixels, GRPH control, swap,
frame counter, flags), u64 staging bytes, plus one native Memory handle. Operation
1 presents the completed staging frame; reply is i64 status plus observed u32
frame counter. Operation 2 restores firmware ownership and replies with i64 status before
closing. Every malformed reply
or timeout stops presentation. The CRTC protocol is private and may change while
the native program-facing APIs are being designed.

Build the updated driver with `driver/gpud/build-musl.sh`. Build the preload with
`driver/client/build.sh` using the dynamic-musl linker wrapper from this crate;
the new additive `BO_IMPORT` operation carries an fd and page-aligned range.
Only existing native Memory mappings can be imported through the shim’s USERPTR
adapter. Arbitrary Linux anonymous pointers are not supported by that adapter.
The mapping must cover Vulkan’s advertised import alignment and allocation size,
as required by [VK_EXT_external_memory_host](https://docs.vulkan.org/spec/latest/chapters/memory.html).

Run the staged sharing proof with the same loader/Mesa/shim/socket environment
as `oscar-vkcheck`, plus `DROSS_BOOTSTRAP=1`:

    oscar-vkshare --bus 2

Use the new gpud and new preload together. Do not use the shared-memory path with
an old server that lacks `BO_IMPORT`. For clocks, `drminfo clockmon OUT` samples
sclk, mclk and temperature four times per second through the normal INFO queries.

For opt-in, authenticated driver/app replacement over SSH, see
[the developer supervisor](../../../driver/gpud/DEVELOPMENT.md). It retains PCI
authority across a drained child exit, verifies release through pcid, and starts
a fresh DMA/display epoch. This avoids rebooting a healthy OS for each build.

The authenticated developer supervisor also passed hardware hot replacement:
with a clock monitor connected, shutdown correctly refused to replace the live
epoch; after that client closed, driver upload/load/restart completed in 1.887 s
without rebooting dross. Missing/wrong authentication tokens were rejected.
Sixteen sequential native Memory import sessions each passed 32 exact GPU
frames and reused the same page-directory VRAM address. Client page tables and
retired display BO ranges are returned only after their completion proofs. The
final import probe also unmapped the original caller mapping and closed its
handle before drawing: 32 frames remained exact through renderer-owned and
independent read mappings. RADV reported a 4096-byte minimum import alignment.
Evidence is `final-v8-alias-probe.txt`, `v6-vm-reuse.txt`, `v6-retire-16.txt`,
`v6-auth-check.txt` and `v6-hot-restart.txt` in the hardware archive.
This validates healthy replacement and repeated client cleanup; it does
not establish recovery from a hung GPU. See the hardware archive for the
supervisor/authentication/restart logs.

After acceptance, `gpud-devctl play` starts a separate interactive native Doom
session with the USB seat and no synthetic input or automatic close. The final
launch is recorded in `results/dross/hardware-20261103/final-v8-play.txt`;
`oscar-doom-indite.png` in that archive shows the real level left open for play.
