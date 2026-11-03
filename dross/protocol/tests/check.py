#!/usr/bin/env python3
"""Build independent C/Rust clients, compare wire bytes and rejected inputs."""
import pathlib, random, struct, subprocess, tempfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
P=ROOT/'protocol'
subprocess.run(['python3',str(P/'generate.py'),'--check'],check=True)
with tempfile.TemporaryDirectory(prefix='oscar-codec-') as tmp:
    c=pathlib.Path(tmp)/'c';r=pathlib.Path(tmp)/'rust'
    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-I'+str(P/'include'),str(P/'tests/c_codec.c'),'-o',str(c)],check=True)
    subprocess.run(['rustc','--edition=2024','-Awarnings',str(P/'tests/rust_codec.rs'),'-o',str(r)],check=True)
    def call(exe,*args):return subprocess.check_output([str(exe),*map(str,args)],text=True).rstrip('\n')
    def check(data,count=0,mode='decode',valid=None):
        a=call(c,mode,data.hex(),count);b=call(r,mode,data.hex(),count)
        assert a==b,(mode,data.hex(),a,b)
        if valid is not None:assert (not a.startswith('-1'))==valid,(data.hex(),a)
    def frame(protocol=0x4f484345,major=1,kind=1,ordinal=1,id=7,flags=0,handles=0,body=b'\x00binary\xff'):
        return struct.pack('<IHHIIQHHI',protocol,major,kind,ordinal,len(body),id,handles,flags,0)+body
    rng=random.Random(0x0d175)
    count=0
    for _ in range(120):
        body=rng.randbytes(rng.randrange(0,100));args=('encode',rng.randrange(1,2**32),rng.randrange(1,65536),1,rng.randrange(2**32),rng.randrange(1,2**64),rng.randrange(65536),rng.randrange(65),body.hex())
        a=call(c,*args);b=call(r,*args);assert a==b;check(bytes.fromhex(a),int(args[-2]),valid=True);count+=1
    good=frame()
    for length in range(len(good)):check(good[:length],valid=False);count+=1
    for at,value in [(0,0),(4,0),(6,0),(6,4),(12,0xffffffff),(16,0),(24,1),(24,65),(28,1)]:
        bad=bytearray(good);size={0:4,4:2,6:2,12:4,16:8,24:2,28:4}[at];bad[at:at+size]=value.to_bytes(size,'little');check(bad,valid=False);count+=1
    for args in [dict(major=2),dict(flags=1),dict(protocol=99),dict(ordinal=999),dict(kind=3,id=0)]:check(frame(**args),valid=True);count+=1
    def record(kind,name,data):return struct.pack('<BBHI',kind,0,len(name),len(data))+name+data
    def boot(records,handles):return frame(0x3254424f,2,3,1,0,0,handles,struct.pack('<HH',len(records),0)+b''.join(records))
    cap=lambda name,i:record(1,name,struct.pack('<IHH',0x5249444f,i,0))
    v=boot([cap(b'directory',0),record(2,b'arg',b'hello')],1);check(v,1,'bootstrap',True);count+=1
    cases=[boot([cap(b'directory',0),cap(b'other',0)],1),boot([cap(b'directory',0),record(2,b'directory',b'')],1),boot([cap(b'directory',1)],1),boot([],1),boot([record(2,b'bad\0',b'')],0),boot([record(2,b'\xc2\x85',b'')],0),boot([record(2,b'\xc0\x80',b'')],0),boot([record(2,b'\xed\xa0\x80',b'')],0),boot([record(2,b'\xf4\x90\x80\x80',b'')],0),boot([record(3,b'name',b'')],0),boot([record(2,b'name',b'x'*1025)],0)]
    for b in cases:check(b,struct.unpack_from('<H',b,24)[0],'bootstrap',False);count+=1
    for name in ['café','日本語','😀','\u00a0']:
        check(boot([record(2,name.encode(),b'ok')],0),0,'bootstrap',True);count+=1
    for length in range(len(v)):check(v[:length],1,'bootstrap',False);count+=1
    print(f'PASS {count} C/Rust framing/bootstrap comparisons; rejected attachments closed')
