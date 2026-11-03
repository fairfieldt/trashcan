use dross_rt::protocol::{datagram as udp,resolver as dns,network,ProtocolError};
fn unhex(s:&str)->Vec<u8>{(0..s.len()).step_by(2).map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap()).collect()}
fn result<T>(r:Result<T,ProtocolError>)->i32{match r{Ok(_)=>0,Err(ProtocolError::Unsupported)=>1,Err(_)=>-1}}
fn main(){let a:Vec<_>=std::env::args().collect();let b=unhex(&a[2]);let h=a.get(3).map(|s|s.parse().unwrap()).unwrap_or(0);let r=match a[1].as_str(){
"udp-request"=>result(udp::decode_request(&b,h)),"dns-request"=>result(dns::decode_request(&b,h)),"packet"=>result(udp::decode_receive_reply(&b,h,3)),"send"=>result(udp::decode_send_reply(&b,h,3)),"answers4"=>result(dns::decode_collect_reply(&b,h,dns::QueryType::A)),"answers6"=>result(dns::decode_collect_reply(&b,h,dns::QueryType::Aaaa)),"query-info"=>result(dns::decode_info(&b)),"name"=>if std::str::from_utf8(&b).is_ok_and(dns::valid_name){0}else{-1},"event"=>result(network::decode_event(&b,h)),_=>panic!("mode")};println!("{r}")}
