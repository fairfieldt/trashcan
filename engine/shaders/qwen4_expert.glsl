// Expert weight formats (specialization constant 0): FMT 0 Q4_0 and FMT 1 IQ4_NL keep each 32-weight block's f16
// scale in a scale array (quant::repack_q4h); FMT 2 IQ4_XS keeps an 8-byte header per 256 weights (f16 d,
// 2 high and 4 low scale bits per 32) in a header array (quant::repack_iq4xs). All three store each 32-weight
// block's nibbles as 4 words from word p.stride: word w holds elements 4w..4w+3 (low nibbles) and 16+4w..
// (high nibbles). IQ4 nibbles index the non-linear IQ4_NL values.
layout(constant_id=0) const uint FMT=0;
shared float lut[16];
const float kv[16]=float[16](-127.,-104.,-83.,-65.,-49.,-35.,-22.,-10.,1.,13.,25.,38.,53.,69.,89.,113.);
// Call before any return: every invocation reaches the barrier.
void init_lut(){if(FMT!=0){if(gl_LocalInvocationID.x<16)lut[gl_LocalInvocationID.x]=kv[gl_LocalInvocationID.x];barrier();}}
float bscale(uint blk){
    if(FMT==2){uint h0=ww.words[(blk>>3)*2],h1=ww.words[(blk>>3)*2+1],ib=blk&7u;
        uint ls=((h1>>(8u*(ib>>1)+4u*(ib&1u)))&15u)|(((h0>>(16u+2u*ib))&3u)<<4);
        return unpackHalf2x16(h0).x*float(int(ls)-32);}
    return unpackHalf2x16(uint(wh.halves[blk])).x;
}
vec4 qlo(uint q){return FMT==0?unpackUnorm4x8(q&0x0f0f0f0fu)*255.0-8.0:vec4(lut[q&15u],lut[(q>>8)&15u],lut[(q>>16)&15u],lut[(q>>24)&15u]);}
vec4 qhi(uint q){return FMT==0?unpackUnorm4x8((q>>4)&0x0f0f0f0fu)*255.0-8.0:vec4(lut[(q>>4)&15u],lut[(q>>12)&15u],lut[(q>>20)&15u],lut[(q>>28)&15u]);}
