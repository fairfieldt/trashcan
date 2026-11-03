use dross_rt::protocol::{network::*,ProtocolError};
use dross_rt::protocol::schema::PROTOCOL_TCP_STREAM;
fn unhex(s:&str)->Vec<u8>{(0..s.len()).step_by(2).map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap()).collect()}
fn hex(b:&[u8]){for v in b{print!("{v:02x}")}println!()}
fn main(){let args:Vec<_>=std::env::args().collect();let b=unhex(&args[2]);let handles=args.get(3).map(|s|s.parse::<usize>().unwrap()).unwrap_or(0);
 match args[1].as_str(){
  "endpoint"=>match Endpoint::decode(&b){Ok(e)=>hex(&e.encode()),Err(_)=>println!("-1")},
  "event"=>match decode_event(&b,handles){Ok(e)=>hex(&encode_event(e).unwrap()),Err(_)=>println!("-1")},
  "info"=>println!("{}",if decode_info(PROTOCOL_TCP_STREAM,&b).is_ok(){0}else{-1}),
  "request"=>println!("{}",match decode_request(&b,handles){Ok(_)=>0,Err(ProtocolError::Unsupported)=>1,Err(_)=>-1}),
  "read"=>println!("{}",if decode_read_reply(&b,handles,16384).is_ok(){0}else{-1}),
  "write"=>println!("{}",if decode_write_reply(&b,handles,16384).is_ok(){0}else{-1}),
  "grant"=>println!("{}",if decode_grant_reply(&b,handles,127).is_ok(){0}else{-1}),
  _=>panic!("mode"),
 }
}
