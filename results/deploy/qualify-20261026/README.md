# Kernel ps9c9a5449 on one Mac, the 4-Mac recipe on a mesh, Thunderbolt recovery (2026-10-26)

Source: main 22c78924 (launcher), engine binary 034690eb... in `~/trashcan-exp/a16` on every Mac (A16's build).
Inventories: [inventory/](inventory/) (the four-Mac mesh, then the six-Mac ring; incite-inflight on bus 2 at both
ends). Every GPU run under gpuq leases; launcher node commands wrapped in `gpuq run ID --`.

## Kernel 7.3.0~rc7-1p2p+ps9c9a5449 on inflight

Package from A16's kbuild build (sha256 f7a0b39c... image, 9f539e13... headers, matching
`results/deploy/a16-20261026/kernel-build/SHA256SUMS`). `install-kernel.sh install --apply`, reboot (back in about
a minute), `verify`: PASS ([logs/inflight-verify.txt](logs/inflight-verify.txt)); `pin --apply` afterwards. The
only new boot messages are amdgpu's missing VCE firmware (video encoder, unused).

| Test | Result |
|---|---|
| 1-gemma smoke | PASS, ready 61 s |
| 1-qwen9b, 1-qwen9b --mtp (model copied from incite with lan-copy) | PASS, ready 19 / 21 s |
| 6-flashnext smoke, chain incite→enlight→invite→indite→alight→inflight | 1st FAIL: answered "France" (2 tokens, 2.5 s); 2nd and 3rd PASS |
| 6-flashnext smoke, chain incite→inflight→alight→indite→invite→enlight | PASS |

The failed run ([logs/six-run1/](logs/six-run1/)): invite's stream to enlight reported "nothing acknowledged for
100ms" at 13:50:05.732 and fell back to TCP; the stream came back at 13:50:06.78 (epoch 795207f2 → 795207f3), as
the smoke request ran (ready at about 13:50:09; the request took 2.5 s against 1-1.4 s otherwise). Both Macs of that
link run 7.3.0-rc7-p2p-s2; inflight's own links stayed on their streams in all four runs. No kernel message on either
Mac. A greedy, seeded request that answers differently once in four runs points at the link's switch to TCP, which
is meant to lose and repeat nothing. Follow-up: TCP fallback is to be a failure, and its cause fixed.

The lights driver for this kernel: `lights/driver.sh build --kernel 7.3.0-rc7-p2p` in kbuild against inflight's
headers (applesmc.ko sha256 0f3c67f9...), `driver.sh install`: 20 port lights, fancurve and portlights-node active.

## 4-qwen27b on a four-Mac full mesh

incite, invite, indite, enlight with all six pairs cabled. Smoke 3 of 3 PASS (ready 46 / 33 / 24 s, requests
0.9-1.5 s), and once with `--mtp` PASS ([logs/four-smoke-*](logs/)). Two links (incite-enlight, invite-enlight) were on
one lane until the all-at-once reload below. A16's two failed first requests of three ran with one link over
Ethernet (`--tcp-fallback`); with every link on Thunderbolt, none failed.

## Thunderbolt recovery ([logs/tb-recovery-20261026.log](logs/tb-recovery-20261026.log))

- `links-reload-all` on all six (mesh cabling): both one-lane links came back on two lanes in one round.
- The inflight-alight link stayed missing through that, and two `scripts/tb-reset.sh inflight alight` rounds: its
  cable had been taken for the mesh. Not a recovery failure.
- Recabled to the six-Mac ring, one `links-reload-all` round: all six links up on two lanes with streams.
