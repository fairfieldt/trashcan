# Recovery-attempt evidence, October 7

These records preserve failed/cancelled attempts and journals around invite's
reboots. They are not successful verification results. In particular,
`invalid-after-reset-tb.json` was produced after a GPU reset and must not be used
for performance or correctness claims. `cancelled-lan.json` is an interrupted
transport test, not Thunderbolt verification. The completed later verification
is recorded separately in the parent evidence directory and commit 460b31b.

The cancellation implementation is still a draft; its current patch, validation
history and missing cross-Mac checks are in `results/engine/shutdown-20261007/`.
