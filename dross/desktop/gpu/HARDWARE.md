# Mac Pro acceptance sequence

Use the target's gpuq lease for every GPU command. Coordinate one-time boot and
Linux recovery with the machine owner. These tests do not change persistent boot
configuration. Keep the running netboot tree unchanged until its replacement is
staged and the next boot is intentional.

1. Run `oscar-vkcheck --bus 6` with the dross Vulkan environment below. Exact
   offscreen pixels exercise graphics, independently of display control.
2. Start the updated gpud and preload together, then run `oscar-vkshare --bus 6`.
   All 32 imported native Memory frames must have exact pixels even after the
   original caller mapping and handle are revoked. Check gpud's
   client-disconnect log before reusing the test slot.
3. Run `oscar-windowcheck` with `windowd-gpu` and the Doom acceptance image.
   This uses GPU composition and firmware framebuffer copying. It checks the
   playable game, held input, focus, occlusion, movement and clean close.
4. Run `oscar-scanoutcheck` with the private scanout channel. It exercises held
   forward, focus, a second native surface and released turn keys, then keeps
   Doom firing visibly for ten seconds before dragging/closing it. Record the
   physical capture, completed-flip count, start/end CRTC counters, windowd's
   restore acknowledgement and gpud's successful firmware restore. The runner
   never reads the original framebuffer for a flipped-image assertion.
5. Run `oscar-vkdraw` as another native-window client with the same Vulkan
   environment. Its GPU destination is the native window Memory; verify its
   contents and pointer response through the compositor. It is a graphics
   application, whereas Doom's game rasterizer is software.

Capture `drminfo clockmon /tmp/oscar-clocks.txt` during sustained GUI work. Save
clocks, exact binary hashes, source snapshot, commands, GPU/CRTC logs and physical
screenshots with the result. A short correctness pass is not a speedup result.
Current mode is retained; other modes/connectors and display interrupts are not
covered by this sequence.

## Environment

The currently installed Indite libraries use these locations. The new preload
must be staged alongside the old preload until the matching new gpud is running.

    LD_LIBRARY_PATH=/disk/alpine-3.22/lib:/disk/alpine-3.22/usr/lib
    LD_PRELOAD=/usr/lib/libgpud-new.so
    VK_DRIVER_FILES=/disk/alpine-3.22/radeon_icd.json
    GPUD_SOCKET=/run/gpud/gpu1.sock
    DROSS_BOOTSTRAP=1

Bus 6 is the display D700 on this machine. Verify the mapping in each boot log.

## Init bindings

In init.conf, `@SOURCE=DESTINATION` consumes a channel end or duplicates an init
capability under the destination's bootstrap name. Geometry comes from the
kernel. Omit an unavailable `display-offset` or supply `%offset=0` for an aligned
firmware address. Keep existing network, USB and disk services.

The checker has its own trusted synthetic seat. It does not receive physical
seat input:

    check oscar-windowcheck 0 @display=display @module/oscar-windowd-gpu=windowd-gpu @module/oscar-doom=doom %width=$display-width %height=$display-height %stride=$display-stride %format=$display-format %offset=0 %gpu-bus=6 %iwad=/disk/.oscar-hardware-20261103/freedoom1.wad %LD_LIBRARY_PATH=/disk/alpine-3.22/lib:/disk/alpine-3.22/usr/lib %LD_PRELOAD=/usr/lib/libgpud-new.so %VK_DRIVER_FILES=/disk/alpine-3.22/radeon_icd.json %GPUD_SOCKET=/run/gpud/gpu1.sock

For bounded private-scanout acceptance, substitute `oscar-scanoutcheck` for
`oscar-windowcheck` and add `@scanout=scanout`; pass the other channel end to gpud
as shown below. Add `@module/oscar-vkdraw=vkdraw` to test the GPU application beside
Doom in the same boot. The Vulkan app verifies a GPU-produced pixel in imported
native Window Memory and reports its rendered pointer position only after the
matching native FrameDone acknowledgement. The runner checks pointer response,
keeps both windows visible during the capture, and requires the app to close
cleanly before releasing the compositor. Current placement assertions require
at least 1230 pixels of screen width for the two-app test.

Build the Doom observer explicitly with
`DOOM_FEATURES=acceptance ./doom/build-musl.sh`. It is used only when the checker
supplies a native `check` channel.

For the interactive session, keep the existing physical seat channel (`seat` in
this example) and add exclusive display handover:

    channel scanout
    start gpud serve 0000:06:00.0 --socket /run/gpud/gpu1.sock @scanout=display
    native oscar-session @display=display @seat=input @scanout=scanout @module/oscar-windowd-gpu=module/oscar-windowd-gpu @module/oscar-doom=module/oscar-doom %app=doom %width=$display-width %height=$display-height %stride=$display-stride %format=$display-format %offset=0 %gpu-bus=6 %iwad=/disk/.oscar-hardware-20261103/freedoom1.wad %LD_LIBRARY_PATH=/disk/alpine-3.22/lib:/disk/alpine-3.22/usr/lib %LD_PRELOAD=/usr/lib/libgpud-new.so %VK_DRIVER_FILES=/disk/alpine-3.22/radeon_icd.json %GPUD_SOCKET=/run/gpud/gpu1.sock

The source channel `scanout` is named `display` inside gpud. The separate physical
`display` Memory goes to the session/windowd and remains reserved for firmware
restore. Do not give the private scanout channel to an application. Remove both
scanout bindings for the first firmware-copy test.

For a Vulkan application session, replace `%app=doom` with `%app=vulkan` and pass
`@module/oscar-vkdraw=module/oscar-vkdraw`. The session supplies the native window
channel and graphics environment; no display capability goes to the application.
To run Doom and Vulkan together, init can create another window channel, pass
one end to the session as `client/vulkan`, and launch vkdraw with the other end
as `window`. Its launcher must set the Linux environment above before execution
(for example an init-launched shell script which exports those values and execs
the staged executable); init bootstrap values alone do not set LD_PRELOAD.
