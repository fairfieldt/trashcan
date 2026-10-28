# MTP through the Responses server, 6-flashnext (2026-10-28)

Why the demo chat ran Flash-Next without MTP, and what MTP gives through the server once on. Six Macs on
7.3.0-rc7-p2p, all six cables up on two lanes, main 4584d9e5 + `mtp_default = true`, 112K context, Q8 K/V, started by
the demo chat's switcher (`tools/webchat/cluster.sh`, gpuq leases on all six).

**The demo of 2026-10-27 had no MTP**: its driver ran `trashcan serve … --ngram disk --nodes …` without `--mtp`
(`deploy/trashcan-cluster up 6-flashnext` without `--mtp`); its reply ran at 20.4 tok/s. Now `6-flashnext` has
`mtp_default = true` and the switcher passes `--mtp` for both models.

**Decode through the server** (`tools/ab.py`: one streamed `/v1/responses` request on incite, decode rate from the
first delta to completion; the server's default sampling is temperature 1.0, top-k 64, top-p 0.95):

| request | without MTP | with MTP | gain |
|---|---|---|---|
| Thunderbolt prose, thinking medium, default sampling (821 tokens) | 21.3, 21.3 | 22.8, 23.2 | +8% |
| the same, temperature 0 (726 tokens) | 21.2, 21.3 | 23.1, 23.4 | +9% |
| 50-state CSV, no thinking, temperature 0 (395 tokens) | 21.3, 21.3 | 26.3 | +23% |

Outputs had the same token counts with and without MTP (a seed gives the same reply). The demo chat's question
("specs of the top of the line Mac Pro 6,1", 3 rounds, `tools/ask.py`) ran at 22.8 tok/s with MTP.

These are well below `qwen4 mtp spec`'s 29.5-33 tok/s on its water/csv/transformer prompts (`../default-112k-20261027/`),
so the server's MTP path gets roughly a third of the benchmark's gain on chat-like text and half on CSV. Not
investigated yet.

**Clocks** (`clocks/`, every second for 15 minutes, both GPUs of each Mac: unix time, GPU, temperature in m°C, sclk in
Hz): at most 61 °C (invite), no sample below 850 MHz while working.
