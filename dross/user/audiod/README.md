# Native audio

`audiod` runs the Intel HDA controller and its analogue codec in a native user process. The kernel supplies PCI authority, memory, DMA isolation, interrupts, channels and deadlines. It contains no audio driver or mixer.

The first version plays finite PCM clips: 48,000 frames per second, two channels, signed 16-bit little-endian samples, left then right. A clip holds at most 48,000 frames. One channel endpoint grants exclusive use of the output. There is no global device path or implicit discovery authority.

Implemented:

- Intel HDA reset, CORB/RIRB codec commands, widget and connection discovery, stereo DAC routing, PCM capability checks, power control, pin control and amplifier setup.
- Private DMA buffers, a three-entry buffer list, MSI completion and bounded stream stop. Clients never get the controller's DMA memory.
- Speaker/headphone selection before playback. The Mac Pro6,1 CS4208 setup includes the board's inverted headphone sense and GPIO 0 speaker enable. These details are gated by codec/subsystem IDs.
- A strict native message protocol and an isolated test client/supervisor.

The controller is selected from Intel PCI audio functions. The two D700 HDMI audio functions are left alone. Codec routes are discovered at boot; CS4208 widget numbers are used only for the vendor setup described in [PROVENANCE.md](PROVENANCE.md).

## Messages and ownership

This is a bringup protocol, not the proposed common public service framing in `API-DESIGN.md`. The protocol is defined in `src/lib.rs`. Every request is exactly 32 bytes, little-endian:

| Offset | Field |
| --- | --- |
| 0 | Version, u32: 1 |
| 4 | Operation, u32: PLAY=1, CLOSE=2, INFO=3 |
| 8 | Request ID, u64 |
| 16 | Absolute monotonic deadline in nanoseconds, u64 |
| 24 | Frame count, u32; zero for INFO/CLOSE |
| 28 | Reserved, u32: zero |

PLAY moves exactly one Memory handle. PCM starts at byte0; the object must cover the page-rounded sample length. Read/map/transfer rights are sufficient. Finish all writes before sending the request. The service maps the object read-only, copies it into private DMA memory, and immediately unmaps and closes the received handle. Later client writes cannot change active playback. The copy uses volatile byte reads without creating ordinary Rust references to externally mutable input. The private copy stays stable during DMA. A client that writes during the copy can produce an inconsistent mixture of old and new samples. Arbitrary PCM bytes are still bounded data: they do not select lengths, pointers or controller operations. This is a snapshot operation, not an atomic read of concurrently modified memory.

Each endpoint accepts one outstanding request. There is no Cancel request or early BufferReleased event. The input is released before playback starts, but the client only learns that at the terminal reply. Closing its endpoint does not immediately cancel accepted playback; the deadline and bounded stop still apply. The reply repeats the request ID and reports the native status, completed frame count and fixed format. On success, playback has stopped. A timeout stops the stream before replying, returns TimedOut and reports zero completed frames; some samples may already have been heard. Invalid messages release their transferred handles. Missing/extra handles, zero/oversized clips and malformed reserved fields fail. INFO returns the format. CLOSE stops the controller, removes DMA mappings, closes interrupts and ends the service.

A deadline must be in the next two seconds. PCI reply waits have a one-second timeout after request serialization; this is not a caller-wide deadline. A lost reply closes and poisons the session, including RELEASE. Timeout does not undo an operation that was already sent. Uncertain DMA remains quarantined until pcid has stopped bus mastering and retired the domain. Codec commands and controller state changes each have a 100 ms bound; codec setup also has a five-second overall limit. A playback deadline includes preparation and the 100 ms silent drain after the clip. Cleanup may need a further 100 ms to confirm that RUN is clear. Silence follows the drain as a guard against immediate ring wrap. This finite-clip API deliberately leaves a gap between requests; continuous low-latency streams and a mixer are future work.

If a codec command loses its reply or DMA stop cannot be confirmed, the driver ends the session instead of reusing ambiguous controller state. It retains the DMA mappings until process teardown. `pcid` owns the domain and disables bus mastering/MSI before dropping it when the client disconnects. A client sees PeerClosed for this fatal failure. There is no automatic replay after a timed-out request.

## Starting a service

The supervisor must provision a `pci` channel to pcid and a `service` channel to the client. Standard init already does this for a native line:

```
channel audio
native audiod @audio=service
native my-audio-client @audio=audio
```

Package the audiod and client modules explicitly. A process launched later through SSH does not acquire PCI authority by knowing the controller's address. The ordinary build, default init configuration and live netboot files are unchanged by the audio work.

## Validation

Run from the repository root:

```
python3 dross/user/audiod/test-qemu.py
python3 dross/user/audiod/test-qemu.py --out dross/target/audio-qemu-smp1 --smp 1
```

The runner uses the tested kernel/pcid/loader snapshot in `dross/target/hardware-memory-final-esp`, builds the new native audio programs and bundled loader, and writes a separate ESP under its output directory. `--esp` selects another tested snapshot. It does not change the standard ESP or netboot selection.

Checks include a lost PCI reply that closes and poisons cloned connections, strict messages, connection-list range expansion, missing handles, an undersized object, an already expired request, an active DMA playback timeout followed by successful reuse, two successful clips, MSI completions, and explicit DMA/interrupt teardown. QEMU writes a WAV. The runner requires both complete clips to appear byte-for-byte in the capture, in order, and checks the left 440 Hz/right 660 Hz signals and level. QEMU's debug-exit bypasses its WAV header cleanup, so the runner finalizes only the two length fields; captured PCM stays unchanged. Logs, command arguments, WAV, metrics, manifest and the self-contained `BOOTX64.EFI` are in the output directory. The same bundled artifact is booted by QEMU and is ready for a later physical acceptance boot.

Validated in QEMU 11.1, Intel HDA 2668, codec 1af40012, SMP 1 and SMP 4. The captured clips contain all 12,000 and 24,000 submitted frames exactly. This proves the emulated controller/DMA/IRQ/IPC path. It does not prove the Mac's analogue amplifier routing or audible physical output.

The prior Linux logs identify INDITE's onboard controller as 00:1b.0 (8086:1d20) and the codec as CS4208. The driver still reads its identity and graph at runtime. Physical playback has not been tested. The dedicated hardware acceptance image must be selected during a coordinated reboot; leave an active Doom session alone.

Remaining: physical CS4208 playback and headphone switching, continuous queued streams with cancellation, volume/mixing policy, device change events, Doom sound integration, recording, resampling, hotplug and HDMI audio.
