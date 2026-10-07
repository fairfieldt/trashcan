# October 7 distributed verification records

`resumed/` contains the completed post-reboot Thunderbolt checks and thermal
measurements recorded in commit 460b31b. `recovery/` preserves earlier interrupted
attempts and kernel evidence; its invalid/cancelled results are not verification.

`scripts/run_case.py` is the original leased, clock-captured runner used for those
attempts and checks. It defaults to all three Thunderbolt stream lanes. Its `--tcp`
option is retained to reproduce earlier transport diagnostics, not as a substitute
for completed Thunderbolt validation. `clippy.log` records the earlier verification
working source’s 19 warnings. Current commit-source compiler validation is recorded
separately in `results/qwen/prefill-slab-20261007/`.
