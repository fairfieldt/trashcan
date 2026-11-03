# Draft native service protocols

Native Oscar protocols carry bytes and capabilities over dross channels. Rust and
C use the same framing and numeric registry. These protocols are drafts; existing
filesystem, graphics and audio bringup protocols remain separate.

`schema.def` assigns protocol IDs, versions, ordinals, application statuses and
limits. `generate.py` produces Rust and C constants. The codecs encode fields
explicitly; no Rust or C structure layout is copied onto the wire. This is a small
registry and codec implementation, not a general interface-definition compiler.

## Framing and ownership

Every message begins with the 32-byte little-endian header in
[Oscar application API design](../API-DESIGN.md#common-service-framing). The body
length and attachment count must match exactly. Request and reply IDs are nonzero;
unsolicited events use zero. Flags and reserved fields are zero in this revision.
Strings are bounded UTF-8; names reject empty strings, NUL and control characters.
Schema limits apply before allocation or attachment access.

The transport owns attached handles until successful receive. Rust's
`owned::Message` then owns them, including during decode failures. Attachments are
referred to by position. A decoder validates the complete message before taking
any attachments. Unknown positions, repeated attachment references, extra handles
and malformed lengths are rejected, closing all unused received handles.

Application statuses beginning `STATUS_` are separate from native syscall errors.
For example, directory `STATUS_DENIED` is not the native error with the same
numeric value. Interpret each in its own result domain.

## Bootstrap v2

Bootstrap uses protocol BOOTSTRAP, major 2, event START and request ID zero. Its
body starts with an entry count (`u16`) and reserved zero (`u16`). Each entry has:

| Field | Encoding |
|---|---|
| Kind | `u8`: 1 for capability, 2 for value |
| Reserved | `u8`, zero |
| Name length | `u16`, UTF-8 byte count |
| Data length | `u32` |
| Name | Exactly the named number of bytes |
| Data | Exactly the declared number of bytes |

Capability data is protocol ID (`u32`), attachment position (`u16`) and reserved
zero (`u16`). Value data is a bounded byte string. Names are unique across both
kinds, and every attachment must be named exactly once. A declared protocol is a
service annotation; it does not replace kernel object-type validation.

Rust `BootstrapV2::take(name)` removes ownership of a capability. A second take
returns None. Dropping the bootstrap closes unused capabilities. `Builder`
preserves a rejected handle in `AddHandleError` and preserves its owned state in
`BuildError` if message construction fails. Legacy text bootstrap is unchanged.

## Directory and provider protocols

A directory connection carries a restricted view of names, protocol IDs and
endpoint rights. Registration uses a separately seeded administrator connection.
All successful connections create fresh service endpoints; the broker releases
its copies after handoff and does not forward service traffic.

| Directory operation | Request body | Successful result after status |
|---|---|---|
| DESCRIBE | Empty | Operation count (`u16`), reserved zero (`u16`), then `u32` ordinals |
| LIST | Empty | Rules for this view |
| CONNECT | Named service and requested rights | Protocol (`u32`), provider instance (`u64`), one endpoint attachment |
| DELEGATE | Requested subset of rules | One new restricted directory attachment |
| REGISTER | Named service and rights ceiling, one provider-control attachment | Empty |

A named-service body contains protocol ID (`u32`), rights (`u32`), name length
(`u16`), reserved zero (`u16`), then the name bytes. A rules body starts with count
(`u16`) and reserved zero (`u16`), followed by named-service records. A reply body
starts with application status (`i32`). Failure replies contain only that status
and no attachments.

Delegation cannot add names, change protocols or add endpoint rights. Registration
cannot be obtained through ordinary lookup or delegation. CONNECT must fit both
the view's rights and the provider's ceiling. Rights govern the returned channel;
the service still enforces its own operations and object authority.

Before returning a connection, the broker sends the provider an OFFER request
containing protocol (`u32`), client endpoint rights (`u32`), instance (`u64`) and
one fresh server endpoint. The provider acknowledges the same request ID with an
application status (`i32`) and no handles. Only a successful acknowledgment
releases the client endpoint to the requester. Replacement affects future
connections; already delivered endpoints continue to identify the old instance.

The Rust directory client has one outstanding call and one reader. IDs increase
for the lifetime of its fresh connection. `NotSent` means no request was queued;
`OutcomeUnknown` means the send succeeded but a reply was not obtained.
`InvalidReply` also retires the local endpoint. An explicit application `Status`
is a known reply. Callers reconnect explicitly and do not automatically replay
operations after an uncertain outcome.

The REGISTER convenience call consumes its provider capability on every outcome,
including failure before sending. Callers needing unsent-capability recovery use
the owned message transport directly. This differs from `try_send`, whose error
always returns the message and its attachments.

The prototype bounds local waits with absolute deadlines. CONNECT does not yet
transmit an operation deadline. The broker independently expires pending handoffs
after two seconds and retires an unresponsive provider control connection. Closing
a client ends its observation and releases its retained client endpoint; it does
not retract an OFFER already delivered to a provider.

## Validation

From the repository root:

```sh
python3 dross/protocol/generate.py --check
python3 dross/protocol/tests/check.py
```

The host checks compare C and Rust bytes and decode results, including truncated
headers, unsupported fields, malformed UTF-8, duplicate names and bad attachment
indices. Native selftests launch `serviced` and a freestanding
[C client](../user/c-api-test/README.md), exercise provider handoff and delegation,
and check cleanup after timeout, provider replacement and client closure.

See [serviced](../user/serviced/README.md) for explicit launch provisioning and
[the Rust SDK](../user/rt/README.md) for transport ownership.
