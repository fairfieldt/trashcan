#!/usr/bin/env python3
"""Compare public UDP/DNS C and Rust codecs on values and hostile bytes."""
from pathlib import Path
import json,random,struct,subprocess,tempfile
root=Path(__file__).resolve().parents[2];p=root/'protocol'
subprocess.run(['python3',str(p/'generate.py'),'--check'],check=True)
with tempfile.TemporaryDirectory(prefix='oscar-network-extensions-') as temp:
 t=Path(temp);(t/'Cargo.toml').write_text('[package]\nname="extension-codec"\nversion="0.0.0"\nedition="2024"\n[dependencies]\ndross-rt={path='+json.dumps(str(root/'user/rt'))+',default-features=false}\n[[bin]]\nname="extension-codec"\npath='+json.dumps(str(p/'tests/extension_codec.rs'))+'\n')
 c=t/'c';subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-I'+str(p/'include'),str(p/'tests/extension_codec.c'),'-o',str(c)],check=True)
 target=root/'target/api-extension-codec';subprocess.run(['cargo','build','--quiet','--manifest-path',str(t/'Cargo.toml'),'--target-dir',str(target)],check=True);rust=target/'debug/extension-codec';cases=0
 def check(mode,data,h=0,expected=None):
  global cases
  args=[mode,data.hex(),str(h)];a=subprocess.check_output([str(c),*args],text=True).strip();b=subprocess.check_output([str(rust),*args],text=True).strip();assert a==b,(mode,data.hex(),h,a,b)
  if expected is not None:assert a==str(expected),(mode,data.hex(),a,expected)
  cases+=1
 def ep(family=4,port=99,scope=0):return struct.pack('<HHI',family,port,scope)+(bytes([127,0,0,1])+bytes(12) if family==4 else bytes.fromhex('00000000000000000000000000000001'))
 def frame(proto,op,body,h=0,kind=1):return struct.pack('<IHHIIQHHI',proto,1,kind,op,len(body),7 if kind!=3 else 0,h,0,0)+body
 deadline=struct.pack('<Q',0xffffffffffffffff);udp=0x5044554f;resolver=0x5345524f;query=0x5152444f
 requests=[('udp-request',frame(0x4b57544e,3,deadline+ep(6,0)+struct.pack('<II',15,0)),0),('udp-request',frame(udp,1,deadline+ep()+struct.pack('<II',0,0)),0),('udp-request',frame(udp,1,deadline+ep(6)+struct.pack('<II',3,0)+b'abc'),0),('udp-request',frame(udp,2,deadline+struct.pack('<II',0,0)),0),('dns-request',frame(resolver,0,deadline),0),('dns-request',frame(resolver,1,deadline+struct.pack('<HHI',1,12,15)+b'example.test'),0),('dns-request',frame(resolver,1,deadline+struct.pack('<HHI',28,1,1)+b'.'),0)]
 for proto,mode in [(udp,'udp-request'),(query,'dns-request')]:
  for op,body,h in [(4,deadline,0),(5,deadline+struct.pack('<IIQ',1 if proto==udp else 64,0,17),1),(6,deadline+struct.pack('<QQ',1,3),0),(7,deadline+struct.pack('<Q',1),0)]:requests.append((mode,frame(proto,op,body,h),h))
 for op in(1,3):requests.append(('dns-request',frame(query,op,deadline),0))
 rng=random.Random(0x554450)
 for mode,data,h in requests:
  check(mode,data,h,0);check(mode,data,1-h,-1)
  for n in range(len(data)):check(mode,data[:n],h,-1)
  for _ in range(20):
   bad=bytearray(data);at=rng.randrange(len(data));bad[at]^=rng.randrange(1,256);check(mode,bad,h)
 def packet(original,copied,truncated):return struct.pack('<IIII',0,truncated,original,copied)+ep(port=0)+ep()+struct.pack('<II',2,0)+b'abc'[:copied]
 values=[('packet',packet(0,0,0)),('packet',packet(3,3,0)),('packet',packet(7,3,1)),('answers4',struct.pack('<IHHIIQ',0,1,1,8,0,9_000_000_000)+ep(port=0)),('answers6',struct.pack('<IHHIIQ',0,28,1,0,0,0)+ep(6,0)),('query-info',struct.pack('<IIIiQQ',0,15,0,0,1,0)),('query-info',struct.pack('<IIIiQQ',0,15,3,-6,2,0)),('query-info',struct.pack('<IIIiQQ',0,15,1,0,3,10)),('event',frame(0x53424f4e,1,struct.pack('<QQQIi',1,4,2,64,0),kind=3)),('send',struct.pack('<II',0,3))]
 for mode,data in values:
  check(mode,data,expected=0)
  for n in range(len(data)):check(mode,data[:n],expected=-1)
  for _ in range(20):
   bad=bytearray(data);at=rng.randrange(len(data));bad[at]^=rng.randrange(1,256);check(mode,bad)
 for mode in ('packet','answers4','answers6'):
  for status in(-2,-4,-6,-8,-9,-16,-17,-18,-19,-20):check(mode,struct.pack('<i',status),expected=0)
 for data,expected in [(b'.',0),(b'example.test.',0),(b'_dns._udp.Example',0),(b'',-1),(b'a..b',-1),(b'a'*64,-1),(b'a.'*128,-1),(b'a\x00b',-1),(b'\xff',-1),(b'a'*63+b'.'+b'b'*63+b'.'+b'c'*63+b'.'+b'd'*63,-1),(b'a'*63+b'.'+b'b'*63+b'.'+b'c'*63+b'.'+b'd'*61,0),(b'a'*63+b'.'+b'b'*63+b'.'+b'c'*63+b'.'+b'd'*61+b'.',0)]:check('name',data,expected=expected)
 check('packet',packet(1,0,0),expected=-1);check('send',struct.pack('<II',0,2),expected=-1)
 print(f'PASS {cases} C/Rust UDP/DNS codec comparisons')
