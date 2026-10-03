# Build before greedy assistant candidate selection

This snapshot contains the unsigned Q4 kernels, adaptive attention, prefix cache and CLI output
change, before the resident assistant began switching between one greedy candidate and 64
stochastic candidates. `assistant.rs` and `server.rs` preserve the two source files changed by
that final adjustment; the remaining engine source matches the parent directory's final build.

The source/binary hashes identify this earlier build. Its six-path GPU reference suite and
2,300-token cache-wrap/rollback check passed. All four HTTP configurations passed their 19-request
prefix checks and both cached configurations passed the Responses integration suite. Exact output
parity is reported separately in each JSON report.

The parent directory's ordinary HTTP reports and `long-check.*` use this build. The final build's
source/binary hashes, reference suite, application benchmarks and speculative HTTP reports live
in the parent directory. Keeping both speculative report sets permits a direct before/after
check of the assistant candidate-count change with the same request sequence.
