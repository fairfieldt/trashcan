# Extraction patches

All patches apply to Fuchsia revision
`650b15d5ea9cfe67faf22ea68cb40e8bc3293303`. They are applied in filename order
by `prepare.py`. The preparation step checks the reverse direction with
`--force --fuzz=0 --dry-run -R`; a missing patch is applied in the forward
direction with the same strict context and no interactive direction guessing.

- `0001-net-declare-without-fidl.patch` removes the unused generated-FIDL common
  type reexport from the address macro wrapper. Packaging omits its FIDL label;
  native address macros remain available. FIDL macros are not supported here.
- `0002-bound-conntrack.patch` replaces the upstream 100,000-entry threshold with
  a strict 1,024-entry budget for the 32-socket worker. A normal flow consumes two
  tuples; a self-connected flow consumes one. The pre-insert capacity check
  accounts for that distinction and repeats existing victim selection until the
  new flow fits. Expired and less-established eviction priority is unchanged;
  established, unexpired flows remain protected. If space cannot be made, the
  existing TableFull packet-drop result is used. The patch also records layout
  sizes through the existing Inspector API and adds mixed-flow unit regressions.

The mixed-flow regressions fill the table with one-entry self flows, advance them
past expiry, then insert two-entry normal flows. Both IPv4 and IPv6 cases failed
the original single-eviction check at 1,025 entries and pass the strict budget.
The complete filter suite now passes 676 tests, including existing full-table
expiry, priority and established-flow protection/drop cases. Evidence is in
`target/mixed-policy-before.log` and `target/filter-policy-tests.log`.

Source provenance:

- [Address macro wrapper](https://fuchsia.googlesource.com/fuchsia/+/650b15d5ea9cfe67faf22ea68cb40e8bc3293303/src/connectivity/lib/net-declare/src/lib.rs)
- [Connection tracking](https://fuchsia.googlesource.com/fuchsia/+/650b15d5ea9cfe67faf22ea68cb40e8bc3293303/src/connectivity/network/netstack3/core/filter/src/conntrack.rs)

The imported source is therefore patched. TCP protocol algorithms are not
rewritten, but the connection-tracking admission policy and diagnostics differ
from upstream. Preserve the Fuchsia copyright headers and root license when
redistributing the imported files.
