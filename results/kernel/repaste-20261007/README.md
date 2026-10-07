# Interrupted post-repaste validation, October 7

These are failure and setup records, not completed post-repaste temperature results.

The first real 27B TP4 corpus run used the f748a035 cancellation candidate, completed
12 corpus rows, stalled in transport and reset both incite GPUs at 11:14. Both device
coredumps were captured and copied to the external workstation. The compressed
buffers, hash metadata, timed clocks and kernel/telemetry logs are in `failure/`.

The repeat used the revised 412c608c drain-worker binary and ended when incite's
GPU PCIe branch went down at 11:31 and the whole system powered off. The final
messages survived in external netconsole (`repeat-external-tail.log`). See
`results/kernel/system-shutdown-20261007/README.md` for the timeline and limits
of the hardware interpretation. The cause of those earlier shutdowns is unresolved.

Both Macs later passed Apple Diagnostics per the user. A subsequent prefill
experiment ended with a user-confirmed circuit-breaker trip; those separate
records are in `results/qwen/prefill-slab-20261007/`. Testing is stopped until the
user confirms the machines are back on another circuit. No aggregate temperature
or successful whole-model candidate throughput is inferred from these failed runs.
