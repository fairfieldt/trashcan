# Service directory

`serviced` is a native userspace broker for the draft
[directory protocol](../../protocol/README.md). It connects explicitly delegated
clients to providers. It has no built-in global service namespace, filesystem root
or device authority.

The launcher sends a BootstrapV2 message with these named capabilities and values:

| Bootstrap entry | Meaning |
|---|---|
| Capability `provider/NAME`, protocol PROVIDER | Provider control endpoint |
| Value `service/NAME` | Service protocol ID as four little-endian bytes |
| Capability `view/LABEL`, protocol DIRECTORY | Server end of a client directory view |
| Value `rules/LABEL` | Encoded rules for that view |
| Capability `admin/LABEL`, protocol DIRECTORY | Separate registration and replacement authority |

At least one client or administrator view is required. Rules specify exact names,
protocols and endpoint-right ceilings. New views can delegate subsets of their
rules; they cannot obtain registration authority. Providers receive a fresh server
endpoint for each successful connection and acknowledge setup before the client
gets its endpoint. Delivered service traffic runs directly between the client and
provider.

The broker permits at most 32 views, eight provider registrations, 32 pending
setups globally, eight pending setups/replies per client admission check, and
eight retained replies per view. Additional admissions fail or close an overloaded
view. Each view contains at most 16 rules. These are bounded prototype limits,
not throughput measurements.

Setup expires after two seconds. A provider that fails to acknowledge is retired;
clients receive an explicit failure and must reconnect or retry deliberately.
Provider replacement changes future connection instance IDs. Existing service
connections are not redirected. Closing all views and draining pending setup ends
the broker. It uses channel notifications with a five-millisecond fallback wake
to retry outbound backpressure.

The ordinary build packages this executable for selftests; init does not yet
install a system-wide directory or migrate existing service startup. Applications
can launch an instance with the native launch helper and the explicit bootstrap
above. The full launcher/supervisor and Linux/POSIX subsystem bridge remain future
work.
