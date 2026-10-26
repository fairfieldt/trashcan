# Port lights

The Mac Pro 6,1 has 20 white LEDs behind its I/O panel. `portlights` (in [`lights/`](../lights/)) uses them as:

- **blinkenlights** while the GPUs work: `portlights-node.service` on each Mac;
- **a music visualizer** across a row of Macs: `portlights conduct` on the machine playing the audio sends each Mac a
  frame 40 times a second.

The engine does not need them.

## Requirements

- The port lights driver: applesmc with kernel patches 0009 and 0010 ([kernel.md](kernel.md)). A kernel built by
  `kernel/build-7.3.sh` with those patches has it. For a kernel without them, build the module for the installed
  kernel ([below](#driver-for-the-installed-kernel)).
- `fancurve.service` ([hardware.md](hardware.md#fan-control)) can keep running; the lights do not change fan control.
- The node runs as root: it reads GPU activity from `/sys/kernel/debug/dri/N/amdgpu_fence_info`.

## Install

On the Mac, in the trashcan checkout:

```sh
cargo build --release -p portlights
sudo lights/install.sh --dry-run          # show what would change
sudo lights/install.sh                    # if the kernel has the driver
sudo lights/install.sh --build-module     # otherwise: build and install the driver too
```

`--build-module` needs `gcc`, `make`, `patch`, `curl` and `linux-headers-$(uname -r)`. To build on another machine,
see the next section and pass the result with `--module applesmc.ko`.

`install.sh` copies the binary to `/usr/local/bin/portlights`, installs and starts `portlights-node.service`, and
writes `/etc/default/portlights`. Running it again changes nothing that is already in place. It never reboots.
`--bin FILE` installs another build (for example one cross-compiled for `x86_64-unknown-linux-gnu`).

Check it:

```sh
systemctl status portlights-node
journalctl -u portlights-node -f          # "blinkenlights" while the GPUs work, "smc" when idle
```

## Driver for the installed kernel

`lights/driver.sh` builds applesmc with 0009 and 0010 for one exact kernel and installs it as
`/lib/modules/$(uname -r)/updates/applesmc.ko`, which takes precedence over the kernel's own applesmc. No reboot.

```sh
lights/driver.sh build                    # on the Mac: for the running kernel; prints the .ko path
sudo lights/driver.sh install lights/build/$(uname -r)/mod/applesmc.ko
lights/driver.sh status
sudo lights/driver.sh uninstall           # back to the kernel's own applesmc
```

- `build` checks that the headers are for exactly the target kernel and downloads that kernel version's
  `applesmc.c` from kernel.org, checked against a pinned SHA-256 (7.3-rc7 and 7.3). For other versions pass
  `--source drivers/hwmon/applesmc.c` from that kernel's source; the build stops if the patches do not apply.
- On another Linux host: `lights/driver.sh build --kernel KREL --headers DIR`, with `DIR` the target's
  `/usr/src/linux-headers-KREL`. Off x86-64 it cross-compiles (`gcc-x86-64-linux-gnu`); add `--host-tools TREE`, a
  kernel source tree of the same version built on that host, so the headers' helper programs run there.
- `install` refuses a module built for another kernel. It reloads applesmc: `fancurve` and `portlights-node` stop for
  a few seconds (the SMC runs the fan meanwhile) and start again. If the new module does not load, the kernel's own
  applesmc is put back.
- Kernels that only load signed modules (Secure Boot) refuse the module.
- After installing a different kernel, build the module again for it, or use a kernel with 0009 and 0010.

## Blinkenlights

The node shows, in order of priority:

1. frames from a conductor, while they keep coming (0.5 s), if the node listens for them;
2. blinkenlights while either GPU signals jobs, and for 10 s after;
3. otherwise the SMC's own behaviour: dark, lit for 15 s when the Mac is moved, the power light breathing.

The USB, Thunderbolt and Ethernet rows count in binary while GPU 0, GPU 1 and the network work; the box edges and HDMI
flicker faster the busier the GPUs are; the audio jacks alternate. Compute work counts, trashcan's or any other job's;
a desktop drawing on the GPUs does not.

Node options go in `PORTLIGHTS_ARGS` in `/etc/default/portlights` (then `sudo systemctl restart portlights-node`):

| Option | Effect |
|---|---|
| `--idle-power on` | the power light steady instead of breathing when idle (`off`, `bright` also work) |
| `--idle off` | the panel dark when idle |
| `--idle blinken` | blinkenlights all the time |
| `--listen 7465` | take frames from a conductor (UDP, all interfaces; `ADDR:7465` for one) |

## Music visualizer

On each Mac, let the node take frames (this opens UDP port 7465):

```sh
sudo lights/install.sh --conductor 7465   # --no-conductor turns it off again
```

On the machine playing the music (Linux or macOS), build `portlights` and list the Macs in
`~/.config/portlights/row` (or `/etc/portlights/row`): one per line, left to right as seen from behind, `-` for a
gap, `#` for comments, `HOST:PORT` for another port. Edits are picked up within 2 s.

```sh
cargo build --release -p portlights
target/release/portlights conduct --style spectrum
```

`--nodes mac1,mac2,...` instead of the row file, `--row FILE` for another file.

| Style | What it shows |
|---|---|
| `spectrum` | one band per Mac, bass on the left; left and right LEDs follow the left and right channels |
| `wave` | a dim spectrum, and a pulse along the row on each kick |
| `meter` | every Mac a stereo level meter |

Audio comes from `--audio CMD`, any command that prints raw 16-bit little-endian stereo at `--rate` (48000):

- Linux, what the machine is playing (the default; PulseAudio or PipeWire):
  `parec --device=@DEFAULT_MONITOR@ --format=s16le --rate=48000 --channels=2 --raw --latency-msec=20`
- A file, paced to real time: `--audio 'ffmpeg -loglevel error -i song.flac -f s16le -ac 2 -ar 48000 -'`
- macOS, playing a file locally while feeding the conductor:
  `--audio 'ffmpeg -loglevel error -i song.flac -f audiotoolbox - -f s16le -ac 2 -ar 48000 -'`

The conductor stops when the audio ends; the Macs go back to blinkenlights or the SMC within 0.5 s.

## Stop and uninstall

```sh
sudo systemctl stop portlights-node       # lights back to the SMC, power light steady
sudo systemctl disable --now portlights-node
sudo lights/uninstall.sh --dry-run
sudo lights/uninstall.sh                  # service, binary, /etc/default/portlights and the driver module
sudo lights/uninstall.sh --keep-driver    # keep the driver module
```

If the node stops or crashes, systemd gives the lights back to the SMC.

## Other commands

```sh
portlights test                  # the LEDs one by one, in SMC order (stop the service first)
portlights blinken --busy 0.8    # blinkenlights without GPU work, 30 s
portlights auto                  # give the lights back to the SMC
```

## Cost

Each changed LED is one SMC transaction (about 0.4 ms). The node skips changes under 3/255.

Measured on Mac Pros with 7.3.0-rc7 and the driver as a module:

- The node uses about 6% of one CPU core during blinkenlights and under 0.5% when idle.
- An SMC read (what `fancurve` does every 2 s) takes 0.4-0.9 ms median with blinkenlights at full speed, against
  0.4 ms without; the slowest of 400 reads was 2.9 ms.
- Six-Mac Qwen3.8-Flash-Next decode (2048-token prompt, 512 tokens): 21.4 and 20.4 tok/s with the node running
  on all six, 20.3 and 20.6 tok/s with it stopped. The difference is within run-to-run variation.

## LED map

From the top of the panel, as seen from behind: the audio jacks (`line-out`, `headphone`), the USB box (`usb-box-top`;
`usb-box-left`, `usb`, `usb-box-right`; `usb-box-bottom`), the Thunderbolt box and the Ethernet box (likewise), `hdmi`,
and the power light (`power`, `power-2`). The driver also exposes each as `/sys/class/leds/smc::NAME`.
