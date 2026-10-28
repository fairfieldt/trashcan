# MTP through the server: the driver's CPU latency (2026-10-28)

Why MTP sped up the Responses server by 8-9% on chat text (`../serve-mtp-20261028/`) when `qwen4 mtp spec` measured
+35-50%: the drafts run on the driver's CPU between GPU steps, and the launcher started the server without the 0 µs CPU
latency request every benchmark ran under (`q4run.py` wraps each process in `scripts/latency-run.py`). Without it,
each round waits for idle cores to leave deep C-states (incite: schedutil, POLL/C1/C1E/C3/C6 all enabled). Plain
decoding barely notices (21.3 -> 21.7 tok/s); the drafting does.

Fix (main 16a91c5e): recipe key `mtp_cpu_latency = true` on `6-flashnext` holds the request for the driver only while
MTP is on (about 55 W more on that Mac while it waits for requests, instead of on all six with the inventory's
`cpu_latency_us = "0"`). The same commit fixes `cpu_latency_us = "0"`, which never started: the launcher passed
`~/trashcan-exp/a16/scripts/latency-run.py` inside the argv after `cd`, where `~` is not expanded.

## Setup

Six Macs on 7.3.0-rc7-p2p, all six cables up on two lanes, engine main 4584d9e5 (`~/trashcan-exp/a16`), 112K context,
Q8 K/V, `--mtp mtp-G-iq4.gguf`, started by the demo chat's switcher (`tools/webchat/cluster.sh`, gpuq leases on all six).
`tools/ab.py`: one streamed `/v1/responses` request on incite; decode rate from the first delta to completion; SHA-256 of
the streamed text. Prose: "Explain how a Thunderbolt 2 cable link works, in about 300 words.", thinking medium, the
server's default sampling (temperature 1.0, top-k 64, top-p 0.95, seed 1). CSV: the 50 states, no thinking,
temperature 0.

## Results (`abba.log`, `tools/abba.sh`)

| run | latency request | prose, 821 tokens | CSV, 395 tokens |
|---|---|---|---|
| A1 | none | 23.6, 23.9 | 27.0, 27.0 |
| B1 | driver | 26.6, 26.7 | 30.3, 30.6 |
| B2 | driver | 26.5, 26.5 | 30.3, 30.2 |
| A2 | none | 23.7, 23.9 | 26.8, 27.1 |
| plain (no MTP) | none | 21.2 | 21.2 |

The driver's request: +12% on both (23.8 -> 26.6, 27.0 -> 30.4). MTP over plain: +25% on prose, +43% on the CSV, in
line with the benchmark. Earlier single runs: the request on all six Macs gave 27.0 / 30.7, on incite alone (held by
hand) 26.7 / 30.4, so the other five Macs don't need it.

**Correctness.** Every run, A, B and plain, streamed the same text: prose `6cbff1b199d6` (sampled), CSV `cab5b86b2aa4`
(greedy). The recipe's smoke with MTP and the driver's request passes (`smoke.log`, `tools/smoke.sh` under leases:
"Paris" in 0.53 s; the driver logs "CPU latency request: 0 us, scoped to this command").

**Clocks** (`clocks/`, every second through the runs; unix time, GPU, m°C, sclk Hz): at most 64 °C (invite), no
working sample below 850 MHz.

Not measured: what separates 26.6 / 30.4 here from the benchmark's 29.5-33 on its own prompts. The server and
`qwen4 mtp spec` run the same `Speculator` per round, so it is most likely how predictable the text is (acceptance per
draft was 0.90-0.95 on the benchmark's prompts; the server doesn't report it).
