# Final TP4 serving validation

`serve-final-v2-plan.json` runs cached plain, fixed MTP, adaptive MTP, and uncached
plain against `trashcan-final`. All four cases enable tiled attention/slab64,
prepared GDN, and direct sum, with `dynamic-smc-maximum` fan metadata. The final
binary restricts direct sum to one-column forwards; these short serving cases
cover plain decode, MTP boundaries, and generated-state cache reuse.

The source archive path `../builds/final/source.tar.gz` is relative to this plan,
resolving to `results/qwen/hardware-20261007/builds/final/source.tar.gz`. The
harness requires that file before either a dry run or real run and records its
SHA256, alongside actual per-host binary hashes. Structural validation before
the build uses a temporary archive fixture; it does not establish build provenance.

After the final build creates the archive, preview exact commands locally from
the repository root:

```sh
python3 results/qwen/hardware-20261007/scripts/cooled-serve-plan.py \
  results/qwen/hardware-20261007/scripts/serve-final-v2-plan.json \
  --out results/qwen/hardware-20261007/plans/serve-final-v2-preview --dry-run
```

After the final binary is deployed to both experiment workspaces, execute:

```sh
python3 results/qwen/hardware-20261007/scripts/cooled-serve-plan.py \
  results/qwen/hardware-20261007/scripts/serve-final-v2-plan.json \
  --out results/qwen/hardware-20261007/plans/serve-final-v2 \
  --max-plan-minutes 30
```

Use unused output paths. The controller waits for all four GPUs to cool to 65 C
before obtaining paired leases. Each case makes 37 bounded requests, releases
only after both processes finish and governors are restored, and compares against
the first cached plain capture. The uncached case requires zero cached tokens and
excludes only that count from response parity. See `SERVE-BUDGET-PLAN.md` for
coverage, health barriers, timeout behavior, and the 14 local mock tests.

The root agent will retarget the existing, not-yet-started long-check-maxfan plan
to the final binary for its 4,300-token check. This plan adds no separate 700-token
GPU check. The historical `serve-budget-plan.json` remains unchanged.

The first final attempt stopped because its client replayed an output item with
`status=incomplete`, which the server rejects as input. The corrected client
preserves the generated text as assistant message input without output lifecycle
metadata. `serve-final-plan.json` and the failed `plans/serve-final` evidence remain
unchanged; the corrected attempt has unique `serve-budget-final-v2-*` run names.

## Audit completed artifacts locally

```sh
python3 results/qwen/hardware-20261007/scripts/audit-final-evidence.py \
  --out results/qwen/hardware-20261007/final-evidence-audit.json
```

The default inputs are `plans/long-check-maxfan-plan` and `plans/serve-final-v2`;
override them with `--long-plan` and `--serve-plan` if their saved locations differ.
Exit 0 means all evidence passed, 1 means an evidence check failed, and 2 means
results are pending or incomplete. This command only reads local artifacts and
writes the requested JSON. It never calls SSH, runs a model, or touches leases.

It checks the final binary and source archive against `builds/final/build.json`,
both completed scopes and restored governors, the long correctness result, all
37 requests in each serving capture, cached/uncached response parity, and successful
release records for each exact host/ticket. Fan identity comes from the original
`fans/incite-deploy.log` and `fans/invite-deploy.log`; later deployments belong in
separate records. Manual fan mode and positive readings must appear in both scope
snapshots. Dynamic maximum/output RPM values remain visible without requiring
exact equality or claiming continuous fan/clock behavior.
