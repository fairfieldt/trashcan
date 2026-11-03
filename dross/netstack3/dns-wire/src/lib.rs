//! Bounded DNS question/answer codec, independent of sockets and Oscar syscalls.
//! Implements standard IN A/AAAA queries (RFC 1035/3596), compression, CNAME
//! following, response correlation and minimum RRset TTL (RFC 2181).
#![no_std]
extern crate alloc;
use alloc::{string::String, vec::Vec};

pub const MAX_MESSAGE: usize = 16_384;
pub const MAX_RECORDS: usize = 64;
pub const MAX_ADDRESSES: usize = 16;
pub const MAX_ALIASES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidName,
    Malformed,
    Unrelated,
    Limit,
    NoMemory,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryType {
    A = 1,
    Aaaa = 28,
}
impl QueryType {
    pub fn from_u16(n: u16) -> Result<Self, Error> {
        match n {
            1 => Ok(Self::A),
            28 => Ok(Self::Aaaa),
            _ => Err(Error::Malformed),
        }
    }
}
/// An expanded wire name, including length bytes and the root terminator.
/// Keeping label boundaries avoids confusing an embedded dot with a separator.
#[derive(Clone, Copy)]
pub struct Name {
    bytes: [u8; 255],
    len: u16,
}
impl core::fmt::Debug for Name {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Name").field(&self.as_wire()).finish()
    }
}
impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        self.as_wire() == other.as_wire()
    }
}
impl Eq for Name {}
impl Name {
    pub const ROOT: Self = Self {
        bytes: [0; 255],
        len: 1,
    };
    pub fn from_ascii(name: &str) -> Result<Self, Error> {
        if name == "." {
            return Ok(Self::ROOT);
        }
        let name = name.strip_suffix('.').unwrap_or(name);
        if name.is_empty() || name.len() > 253 {
            return Err(Error::InvalidName);
        }
        let mut n = Self::ROOT;
        let mut at = 0;
        for label in name.as_bytes().split(|b| *b == b'.') {
            if label.is_empty() || label.len() > 63 || !label.iter().all(|b| host_byte(*b)) {
                return Err(Error::InvalidName);
            }
            if at + 1 + label.len() + 1 > 255 {
                return Err(Error::InvalidName);
            }
            n.bytes[at] = label.len() as u8;
            at += 1;
            for b in label {
                n.bytes[at] = b.to_ascii_lowercase();
                at += 1;
            }
        }
        n.bytes[at] = 0;
        n.len = (at + 1) as u16;
        Ok(n)
    }
    pub fn as_wire(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
    pub fn to_ascii(&self) -> Result<String, Error> {
        let mut text = String::new();
        text.try_reserve_exact(self.len as usize)
            .map_err(|_| Error::NoMemory)?;
        let mut at = 0;
        while self.bytes[at] != 0 {
            let len = self.bytes[at] as usize;
            at += 1;
            if !text.is_empty() {
                text.push('.')
            }
            for b in &self.bytes[at..at + len] {
                if !host_byte(*b) {
                    return Err(Error::InvalidName);
                }
                text.push(*b as char);
            }
            at += len;
        }
        if text.is_empty() {
            text.push('.')
        }
        Ok(text)
    }
}
fn host_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    V4([u8; 4]),
    V6([u8; 16]),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    pub canonical_name: Name,
    pub ttl: u32,
    pub addresses: Vec<Address>,
    pub aliases: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Response {
    Answer(Answer),
    Alias { name: Name, ttl: u32, hops: usize },
    NameNotFound,
    NoData,
    TemporaryFailure,
    Truncated,
}
#[derive(Clone, Copy, Debug)]
pub struct Question {
    pub id: u16,
    pub name: Name,
    pub kind: QueryType,
}
impl Question {
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        let mut b = Vec::new();
        b.try_reserve_exact(16 + self.name.len as usize)
            .map_err(|_| Error::NoMemory)?;
        b.extend_from_slice(&self.id.to_be_bytes());
        b.extend_from_slice(&0x0100u16.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&[0; 6]);
        b.extend_from_slice(self.name.as_wire());
        b.extend_from_slice(&(self.kind as u16).to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        Ok(b)
    }
}
fn u16_at(b: &[u8], at: usize) -> Result<u16, Error> {
    let s = b.get(at..at + 2).ok_or(Error::Malformed)?;
    Ok(u16::from_be_bytes([s[0], s[1]]))
}
fn u32_at(b: &[u8], at: usize) -> Result<u32, Error> {
    let s = b.get(at..at + 4).ok_or(Error::Malformed)?;
    Ok(u32::from_be_bytes(s.try_into().unwrap()))
}
/// Advance the caller only across the compressed name's on-wire bytes.
fn name_at(message: &[u8], position: &mut usize) -> Result<Name, Error> {
    let mut out = Name::ROOT;
    let mut written = 0;
    let mut at = *position;
    let mut consumed = None;
    let mut targets = [usize::MAX; 32];
    let mut jumps = 0;
    for _ in 0..256 {
        let size = *message.get(at).ok_or(Error::Malformed)?;
        if size & 0xc0 == 0xc0 {
            let pair = u16_at(message, at)?;
            let target = (pair & 0x3fff) as usize;
            if target < 12 || target >= at || targets[..jumps].contains(&target) {
                return Err(Error::Malformed);
            }
            if jumps == targets.len() {
                return Err(Error::Limit);
            }
            targets[jumps] = target;
            jumps += 1;
            if consumed.is_none() {
                consumed = Some(at + 2)
            }
            at = target;
            continue;
        }
        if size & 0xc0 != 0 {
            return Err(Error::Malformed);
        }
        at += 1;
        if size == 0 {
            if written >= 255 {
                return Err(Error::Malformed);
            }
            out.bytes[written] = 0;
            out.len = (written + 1) as u16;
            *position = consumed.unwrap_or(at);
            return Ok(out);
        }
        let label = message
            .get(at..at + size as usize)
            .ok_or(Error::Malformed)?;
        if written + size as usize + 2 > 255 {
            return Err(Error::Malformed);
        }
        out.bytes[written] = size;
        written += 1;
        for b in label {
            out.bytes[written] = b.to_ascii_lowercase();
            written += 1;
        }
        at += size as usize;
    }
    Err(Error::Limit)
}
#[derive(Clone, Debug)]
enum Data {
    Address(Address),
    Alias(Name),
    Other,
}
#[derive(Clone, Debug)]
struct Record {
    owner: Name,
    ttl: u32,
    data: Data,
}
fn normalized_ttl(raw: u32) -> u32 {
    if raw & 0x8000_0000 != 0 { 0 } else { raw }
}

/// Validate the complete reply before exposing answer data. UDP callers discard
/// Unrelated/malformed packets while their original deadline remains in force.
pub fn parse(message: &[u8], question: &Question) -> Result<Response, Error> {
    if message.len() < 12 {
        return Err(Error::Malformed);
    }
    if message.len() > MAX_MESSAGE {
        return Err(Error::Limit);
    }
    if u16_at(message, 0)? != question.id {
        return Err(Error::Unrelated);
    }
    let flags = u16_at(message, 2)?;
    if flags & 0x8000 == 0 || flags & 0x7800 != 0 || flags & 0x0040 != 0 {
        return Err(Error::Malformed);
    }
    if u16_at(message, 4)? != 1 {
        return Err(Error::Malformed);
    }
    let counts = [
        u16_at(message, 6)? as usize,
        u16_at(message, 8)? as usize,
        u16_at(message, 10)? as usize,
    ];
    let mut at = 12;
    let name = name_at(message, &mut at)?;
    if name != question.name
        || u16_at(message, at)? != question.kind as u16
        || u16_at(message, at + 2)? != 1
    {
        return Err(Error::Unrelated);
    }
    at += 4;
    if flags & 0x0200 != 0 {
        return Ok(Response::Truncated);
    }
    if counts.iter().sum::<usize>() > MAX_RECORDS {
        return Err(Error::Limit);
    }
    let mut answers = Vec::new();
    answers
        .try_reserve_exact(counts[0])
        .map_err(|_| Error::NoMemory)?;
    for section in 0..3 {
        for _ in 0..counts[section] {
            let owner = name_at(message, &mut at)?;
            let kind = u16_at(message, at)?;
            let class = u16_at(message, at + 2)?;
            let ttl = normalized_ttl(u32_at(message, at + 4)?);
            let len = u16_at(message, at + 8)? as usize;
            at += 10;
            let end = at.checked_add(len).ok_or(Error::Malformed)?;
            let rdata = message.get(at..end).ok_or(Error::Malformed)?;
            let data = if class == 1 {
                match kind {
                    1 => {
                        if len != 4 {
                            return Err(Error::Malformed);
                        }
                        Data::Address(Address::V4(rdata.try_into().unwrap()))
                    }
                    28 => {
                        if len != 16 {
                            return Err(Error::Malformed);
                        }
                        Data::Address(Address::V6(rdata.try_into().unwrap()))
                    }
                    5 => {
                        let mut p = at;
                        let target = name_at(message, &mut p)?;
                        if p != end {
                            return Err(Error::Malformed);
                        }
                        Data::Alias(target)
                    }
                    _ => Data::Other,
                }
            } else {
                Data::Other
            };
            if section == 0 {
                answers.push(Record { owner, ttl, data })
            }
            at = end;
        }
    }
    if at != message.len() {
        return Err(Error::Malformed);
    }
    match flags & 15 {
        0 => {}
        3 => return Ok(Response::NameNotFound),
        1 | 2 | 4 | 5 => return Ok(Response::TemporaryFailure),
        _ => return Err(Error::Malformed),
    }
    let mut current = question.name;
    let mut seen = [Name::ROOT; MAX_ALIASES + 1];
    seen[0] = current;
    let mut hops = 0;
    let mut ttl = u32::MAX;
    loop {
        let mut target = None;
        let mut alias_ttl = u32::MAX;
        let mut has_address = false;
        for record in &answers {
            if record.owner == current {
                match record.data {
                    Data::Alias(name) => {
                        if target.is_some_and(|old| old != name) {
                            return Err(Error::Malformed);
                        }
                        target = Some(name);
                        alias_ttl = alias_ttl.min(record.ttl)
                    }
                    Data::Address(_) => has_address = true,
                    Data::Other => {}
                }
            }
        }
        if let Some(next) = target {
            if has_address || seen[..=hops].contains(&next) {
                return Err(Error::Malformed);
            }
            if hops == MAX_ALIASES {
                return Err(Error::Limit);
            }
            next.to_ascii()?;
            hops += 1;
            seen[hops] = next;
            current = next;
            ttl = ttl.min(alias_ttl);
            continue;
        }
        let mut addresses = Vec::new();
        addresses
            .try_reserve_exact(MAX_ADDRESSES)
            .map_err(|_| Error::NoMemory)?;
        for record in &answers {
            if record.owner == current {
                if let Data::Address(address) = record.data {
                    if matches!(
                        (question.kind, address),
                        (QueryType::A, Address::V4(_)) | (QueryType::Aaaa, Address::V6(_))
                    ) {
                        ttl = ttl.min(record.ttl);
                        if !addresses.contains(&address) {
                            if addresses.len() == MAX_ADDRESSES {
                                return Err(Error::Limit);
                            }
                            addresses.push(address);
                        }
                    }
                }
            }
        }
        if !addresses.is_empty() {
            return Ok(Response::Answer(Answer {
                canonical_name: current,
                ttl,
                addresses,
                aliases: hops,
            }));
        }
        if hops > 0 {
            return Ok(Response::Alias {
                name: current,
                ttl,
                hops,
            });
        }
        // A referral from a non-recursive, non-authoritative server is not a
        // negative answer. Never follow untrusted additional-section glue.
        if flags & 0x0480 == 0 {
            return Ok(Response::TemporaryFailure);
        }
        return Ok(Response::NoData);
    }
}

/// Decode one standard query for an explicitly provisioned DNS test/server
/// endpoint. This does not grant authority to answer any other query.
pub fn decode_question(message: &[u8]) -> Result<Question, Error> {
    if message.len() < 12 || message.len() > MAX_MESSAGE {
        return Err(Error::Malformed);
    }
    if u16_at(message, 2)? & 0xf840 != 0
        || u16_at(message, 4)? != 1
        || u16_at(message, 6)? != 0
        || u16_at(message, 8)? != 0
        || u16_at(message, 10)? != 0
    {
        return Err(Error::Malformed);
    }
    let mut at = 12;
    let name = name_at(message, &mut at)?;
    let kind = QueryType::from_u16(u16_at(message, at)?)?;
    if u16_at(message, at + 2)? != 1 || at + 4 != message.len() {
        return Err(Error::Malformed);
    }
    Ok(Question {
        id: u16_at(message, 0)?,
        name,
        kind,
    })
}
#[derive(Clone, Copy, Debug)]
pub enum RecordData {
    A([u8; 4]),
    Aaaa([u8; 16]),
    Cname(Name),
}
#[derive(Clone, Copy, Debug)]
pub struct ResourceRecord {
    pub name: Name,
    pub ttl: u32,
    pub data: RecordData,
}
/// Small response encoder used by deterministic server fixtures. No compression
/// is necessary in generated packets; the parser also accepts compressed names.
pub fn encode_response(
    question: &Question,
    rcode: u8,
    truncated: bool,
    records: &[ResourceRecord],
) -> Result<Vec<u8>, Error> {
    if rcode > 15 || records.len() > MAX_RECORDS {
        return Err(Error::Limit);
    }
    let mut size = 16 + question.name.len as usize;
    for rr in records {
        size += rr.name.len as usize
            + 10
            + match rr.data {
                RecordData::A(_) => 4,
                RecordData::Aaaa(_) => 16,
                RecordData::Cname(n) => n.len as usize,
            };
    }
    if size > MAX_MESSAGE {
        return Err(Error::Limit);
    }
    let mut b = Vec::new();
    b.try_reserve_exact(size).map_err(|_| Error::NoMemory)?;
    b.extend_from_slice(&question.id.to_be_bytes());
    let flags = 0x8180u16 | if truncated { 0x0200 } else { 0 } | rcode as u16;
    b.extend_from_slice(&flags.to_be_bytes());
    b.extend_from_slice(&1u16.to_be_bytes());
    b.extend_from_slice(&(records.len() as u16).to_be_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(question.name.as_wire());
    b.extend_from_slice(&(question.kind as u16).to_be_bytes());
    b.extend_from_slice(&1u16.to_be_bytes());
    for rr in records {
        b.extend_from_slice(rr.name.as_wire());
        let (kind, rdata): (u16, &[u8]) = match &rr.data {
            RecordData::A(a) => (1, a),
            RecordData::Aaaa(a) => (28, a),
            RecordData::Cname(n) => (5, n.as_wire()),
        };
        b.extend_from_slice(&kind.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&rr.ttl.to_be_bytes());
        b.extend_from_slice(&(rdata.len() as u16).to_be_bytes());
        b.extend_from_slice(rdata);
    }
    Ok(b)
}

#[cfg(test)]
extern crate std;
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    fn q() -> Question {
        Question {
            id: 0x7642,
            name: Name::from_ascii("Oscar.Example.").unwrap(),
            kind: QueryType::A,
        }
    }
    fn record(n: &str, ttl: u32, data: RecordData) -> ResourceRecord {
        ResourceRecord {
            name: Name::from_ascii(n).unwrap(),
            ttl,
            data,
        }
    }
    fn answer(response: Response) -> Answer {
        match response {
            Response::Answer(a) => a,
            _ => panic!("missing answer"),
        }
    }
    #[test]
    fn actual_query_wire_and_name_limits() {
        let question = q();
        let b = question.encode().unwrap();
        assert_eq!(&b[..12], &[0x76, 0x42, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&b[12..], b"\x05oscar\x07example\0\0\x01\0\x01");
        assert_eq!(decode_question(&b).unwrap().name, question.name);
        let max = format!(
            "{}.{}.{}.{}.",
            "a".repeat(63),
            "b".repeat(63),
            "c".repeat(63),
            "d".repeat(61)
        );
        assert_eq!(Name::from_ascii(&max).unwrap().as_wire().len(), 255);
        for invalid in ["", "a..b", ".a", "a..", "a b", "a\0b", "café"] {
            assert_eq!(Name::from_ascii(invalid), Err(Error::InvalidName));
        }
        assert!(Name::from_ascii(&"a".repeat(64)).is_err());
        assert_eq!(Name::from_ascii(".").unwrap().to_ascii().unwrap(), ".");
    }
    #[test]
    fn compressed_rrsets_correlation_and_ttl() {
        let question = q();
        let mut b = question.encode().unwrap();
        b[2..4].copy_from_slice(&0x8180u16.to_be_bytes());
        b[6..8].copy_from_slice(&2u16.to_be_bytes());
        for (ttl, address) in [(90u32, [192, 0, 2, 1]), (30, [192, 0, 2, 2])] {
            b.extend_from_slice(&[0xc0, 0x0c, 0, 1, 0, 1]);
            b.extend_from_slice(&ttl.to_be_bytes());
            b.extend_from_slice(&[0, 4]);
            b.extend_from_slice(&address);
        }
        let a = answer(parse(&b, &question).unwrap());
        assert_eq!(a.ttl, 30);
        assert_eq!(a.addresses.len(), 2);
        let mut other = question;
        other.id += 1;
        assert_eq!(parse(&b, &other), Err(Error::Unrelated));
        other = question;
        other.name = Name::from_ascii("other.example").unwrap();
        assert_eq!(parse(&b, &other), Err(Error::Unrelated));
        for n in 0..b.len() {
            assert!(parse(&b[..n], &question).is_err(), "truncation {n}");
        }
        b.push(0);
        assert_eq!(parse(&b, &question), Err(Error::Malformed));
    }
    #[test]
    fn unordered_cname_and_unrelated_additional_data() {
        let question = q();
        let target = Name::from_ascii("final.example").unwrap();
        let records = [
            record("final.example", 60, RecordData::A([192, 0, 2, 8])),
            record("unrelated.example", 1, RecordData::A([203, 0, 113, 1])),
            record("oscar.example", 10, RecordData::Cname(target)),
        ];
        let b = encode_response(&question, 0, false, &records).unwrap();
        let a = answer(parse(&b, &question).unwrap());
        assert_eq!(a.canonical_name, target);
        assert_eq!(a.ttl, 10);
        assert_eq!(a.addresses, vec![Address::V4([192, 0, 2, 8])]);
        let b = encode_response(&question, 0, false, &records[..1]).unwrap();
        assert_eq!(parse(&b, &question), Ok(Response::NoData));
        let b = encode_response(&question, 0, false, &records[2..]).unwrap();
        assert_eq!(
            parse(&b, &question),
            Ok(Response::Alias {
                name: target,
                ttl: 10,
                hops: 1
            })
        );
    }
    #[test]
    fn compression_loops_bad_lengths_and_rrset_limits() {
        let question = q();
        let mut b = encode_response(
            &question,
            0,
            false,
            &[record("oscar.example", 10, RecordData::A([1, 2, 3, 4]))],
        )
        .unwrap();
        let at = question.encode().unwrap().len();
        b[at] = 0xc0;
        b[at + 1] = at as u8;
        assert!(parse(&b, &question).is_err());
        let records = [record(
            "oscar.example",
            10,
            RecordData::Cname(question.name),
        )];
        assert_eq!(
            parse(
                &encode_response(&question, 0, false, &records).unwrap(),
                &question
            ),
            Err(Error::Malformed)
        );
        let mut records = Vec::new();
        for n in 0..=MAX_ADDRESSES {
            records.push(record(
                "oscar.example",
                30,
                RecordData::A([192, 0, 2, n as u8]),
            ));
        }
        assert_eq!(
            parse(
                &encode_response(&question, 0, false, &records).unwrap(),
                &question
            ),
            Err(Error::Limit)
        );
    }
    #[test]
    fn aaaa_negative_truncation_and_high_ttl() {
        let mut question = q();
        question.kind = QueryType::Aaaa;
        let mut address = [0; 16];
        address[0] = 0x20;
        address[1] = 1;
        address[15] = 7;
        let record = record("oscar.example", 0x8000_0001, RecordData::Aaaa(address));
        let a = answer(
            parse(
                &encode_response(&question, 0, false, &[record]).unwrap(),
                &question,
            )
            .unwrap(),
        );
        assert_eq!(a.ttl, 0);
        assert_eq!(a.addresses, vec![Address::V6(address)]);
        for (rcode, response) in [
            (3, Response::NameNotFound),
            (2, Response::TemporaryFailure),
            (0, Response::NoData),
        ] {
            assert_eq!(
                parse(
                    &encode_response(&question, rcode, false, &[]).unwrap(),
                    &question
                ),
                Ok(response)
            );
        }
        let mut b = encode_response(&question, 0, true, &[]).unwrap();
        b[6..8].copy_from_slice(&100u16.to_be_bytes());
        assert_eq!(parse(&b, &question), Ok(Response::Truncated));
    }
    #[test]
    fn deterministic_mutations_never_panic() {
        let question = q();
        let packet = encode_response(
            &question,
            0,
            false,
            &[record("oscar.example", 30, RecordData::A([1, 2, 3, 4]))],
        )
        .unwrap();
        for at in 0..packet.len() {
            for value in [0, 1, 63, 64, 128, 192, 255] {
                let mut p = packet.clone();
                p[at] = value;
                let _ = parse(&p, &question);
            }
        }
    }
}
