# DNS forwarding resolver

`oscar-resolver` runs bounded DNS jobs on the shared network engine. The daemon
supplies one trusted recursive server. Clients cannot select an upstream through
a lookup request. There are no ambient resolver files, search suffixes or shared
answer cache. A and AAAA are separate query kinds.

`start` admits one of eight jobs. `pump` advances local work after Ethernet input,
engine progress or a real deadline; it never blocks or polls on a fixed interval.
UDP attempts use fresh secure transaction IDs per question and randomized source
ports. Retries wait two and then four seconds. An active job lasts at most ten
seconds and always respects the caller's earlier deadline.

Replies must come from the configured address and port and match the question.
Unrelated or malformed UDP traffic is discarded within the same deadline.
Truncated UDP answers switch to length-prefixed TCP, including partial writes
and reads. Invalid TCP responses end the job. CNAME continuations use the same
configured server and original deadline; a chain may have at most eight links.

A successful result holds up to 16 addresses and a fixed minimum absolute
expiry. A later CNAME response cannot refresh an earlier link's TTL. Repeated
collection keeps that expiry. Completion releases request buffers and sockets;
the small result remains until the query capability closes. Cancel stops future
I/O and retains Canceled state for inspection. Packets already accepted cannot
be recalled.

The facade uses internal engine IDs. Its trusted owner must call `remove` when
a client closes, or `shutdown` when retiring the facade; these cancel active
transport work. The daemon implements this ownership boundary.

`resolver-test` is a native Oscar executable. Its simulated local server checks
A/AAAA, foreign sender/ID/name rejection, packet-loss retry, partial TCP fallback,
CNAME expiry, negative results, cancellation, timeout and admission limits.
It uses real Oscar entropy and virtual network time. See the parent validation
record for the tested native images and physical service checks.

This is a forwarding resolver and does not independently validate DNSSEC.
TCP behavior follows [RFC 7766](https://www.rfc-editor.org/rfc/rfc7766).
