#!/usr/bin/env python3
"""Compare the public C/Rust native TCP codecs on valid and hostile wire input."""
from pathlib import Path
import json,random,struct,subprocess,tempfile
root=Path(__file__).resolve().parents[2]; protocol=root/'protocol'
subprocess.run(['python3',str(protocol/'generate.py'),'--check'],check=True)
with tempfile.TemporaryDirectory(prefix='oscar-network-codec-') as tmp:
    tmp=Path(tmp); c=tmp/'c'; build=tmp/'rust';build.mkdir()
    (build/'Cargo.toml').write_text('[package]\nname="network-codec"\nversion="0.0.0"\nedition="2024"\n[dependencies]\ndross-rt={path='+json.dumps(str(root/'user/rt'))+',default-features=false}\n[[bin]]\nname="network-codec"\npath='+json.dumps(str(protocol/'tests/network_codec.rs'))+'\n')
    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-I'+str(protocol/'include'),str(protocol/'tests/network_codec.c'),'-o',str(c)],check=True)
    target=root/'target/api-network-codec'
    subprocess.run(['cargo','build','--quiet','--manifest-path',str(build/'Cargo.toml'),'--target-dir',str(target)],check=True)
    rust=target/'debug/network-codec';cases=0
    def check(mode,data,handles=0,expected=None):
        global cases
        args=[mode,data.hex(),str(handles)]
        a=subprocess.check_output([str(c),*args],text=True).strip();b=subprocess.check_output([str(rust),*args],text=True).strip()
        assert a==b,(mode,data.hex(),handles,a,b)
        if expected is not None:assert a==expected,(mode,data.hex(),a,expected)
        cases+=1
    def frame(proto,op,body,handles=0,kind=1,id=7):return struct.pack('<IHHIIQHHI',proto,1,kind,op,len(body),id,handles,0,0)+body
    v4=struct.pack('<HHI',4,443,0)+bytes([127,0,0,1])+bytes(12)
    v6=struct.pack('<HHI',6,80,3)+bytes.fromhex('fe800000000000000000000000000001')
    for endpoint in (v4,v6):
        check('endpoint',endpoint,expected=endpoint.hex())
        for n in range(24):check('endpoint',endpoint[:n],expected='-1')
    for offset in (0,1,4,12,23):
        bad=bytearray(v4);bad[offset]^=1;check('endpoint',bad,expected='-1')
    event=frame(0x53424f4e,1,struct.pack('<QQQIi',9,17,4,2,0),kind=3,id=0)
    check('event',event,expected=event.hex())
    for n in range(len(event)):check('event',event[:n],expected='-1')
    check('event',event,1,expected='-1')
    for offset in (24,28,32,48,56,60):
        bad=bytearray(event);bad[offset]^=1;check('event',bad)
    info=struct.pack('<IIIiQ',0,127,4,0,1)+v4+v6+struct.pack('<II',3,0)
    check('info',info,expected='0')
    for n in range(80):check('info',info[:n],expected='-1')
    for offset in (0,4,8,12,16,24,28,36,48,72,76):
        bad=bytearray(info);bad[offset]^=128;check('info',bad)
    deadline=struct.pack('<Q',0xffffffffffffffff)
    requests=[(0x4b57544e,0,deadline,0),(0x4b57544e,1,deadline+v4+struct.pack('<II',127,0),0),(0x4b57544e,1,deadline+v6+struct.pack('<II',1,0),0),(0x4b57544e,2,deadline+v4+struct.pack('<IIII',7,127,4,0),0),(0x4c504354,1,deadline+struct.pack('<II',1,0),0),(0x53504354,1,deadline+struct.pack('<II',16384,0),0),(0x53504354,2,deadline+struct.pack('<II',3,0)+b'abc',0)]
    for proto in (0x53504354,0x4c504354):
        requests.extend([(proto,4,deadline,0),(proto,5,deadline+struct.pack('<IIQ',1 if proto==0x53504354 else 8,0,6),1),(proto,6,deadline+struct.pack('<QQ',2,3),0),(proto,7,deadline+struct.pack('<Q',2),0)])
    for op in (3,8,9):requests.append((0x53504354,op,deadline,0))
    rng=random.Random(0x4e4554)
    for proto,op,body,handles in requests:
        packet=frame(proto,op,body,handles);check('request',packet,handles,expected='0');check('request',packet,1-handles,expected='-1')
        for n in range(len(packet)):check('request',packet[:n],handles,expected='-1')
        for _ in range(12):
            bad=bytearray(packet);at=rng.randrange(len(packet));bad[at]^=rng.randrange(1,256);check('request',bad,handles)
    for mode,bodies in [('read',[struct.pack('<IIII',0,0,3,0)+b'abc',struct.pack('<IIII',0,1,0,0),struct.pack('<i',-9),struct.pack('<IIII',0,0,0,0),struct.pack('<IIII',0,1,1,0)+b'x']),('write',[struct.pack('<II',0,7),struct.pack('<i',-9),struct.pack('<II',0,0),struct.pack('<II',0,16385)]),('grant',[struct.pack('<II',0,127),struct.pack('<i',-2),struct.pack('<II',0,128)])]:
        for body in bodies:
            handles=1 if mode=='grant' and len(body)==8 else 0;check(mode,body,handles)
            for n in range(len(body)):check(mode,body[:n],handles,expected='-1')
    print(f'PASS {cases} C/Rust native-network codec comparisons')
