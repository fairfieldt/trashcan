# DNS message codec

This `no_std` crate encodes IN A/AAAA questions and validates DNS responses. It
contains no sockets, service lookup or clock access. Allocation is fallible and
bounded to 16 KiB messages, 64 resource records, 16 address results and eight
CNAME links.

Names keep their expanded wire label boundaries. Compression pointers must point
backward inside the message; bounded traversal rejects cycles, invalid lengths
and reserved label encodings. A reply must match its transaction ID, question,
class and type. Only answer-section records on the CNAME chain supply addresses;
additional-section glue never redirects a query or supplies an answer.

Results distinguish a complete address set, CNAME continuation, NXDOMAIN,
no data, temporary failure and UDP truncation. TTL is the minimum across the
answer set and CNAMEs. High-bit TTL values become zero. The transport layer must
also validate the sender and carry an absolute expiry across follow-up queries.

`decode_question` and `encode_response` support deterministic DNS server fixtures.
The host suite covers compressed and reordered records, hostile lengths and
pointers, unrelated replies, A/AAAA, TTLs, negative results and bounded mutations:

```sh
cargo test -p oscar-dns-wire --locked
```

Protocol references: [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035),
[RFC 3596](https://www.rfc-editor.org/rfc/rfc3596),
[RFC 2181](https://www.rfc-editor.org/rfc/rfc2181).
