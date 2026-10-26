import struct, sys, glob, re
T={0:'F32',1:'F16',2:'Q4_0',8:'Q8_0',14:'Q6_K',30:'BF16'}
def rd(f,fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))
def s(f): n,=rd(f,'Q'); return f.read(n).decode()
def skip(f,t):
    sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t==8: s(f)
    elif t==9:
        et,n=rd(f,'IQ')
        for _ in range(n): skip(f,et)
    else: f.read(sz[t])
for p in sorted(glob.glob(sys.argv[1])):
    with open(p,'rb') as f:
        assert f.read(4)==b'GGUF'; v,nt,nk=rd(f,'IQQ')
        for _ in range(nk): s(f); t,=rd(f,'I'); skip(f,t)
        for _ in range(nt):
            n=s(f); nd,=rd(f,'I'); dims=rd(f,'Q'*nd); t,off=rd(f,'IQ')
            if re.search(sys.argv[2], n): print(n, T.get(t,t), dims)
