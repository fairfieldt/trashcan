// Formats and math checked against llama.cpp; see LICENSE.llama.cpp for its MIT notice.
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(binding=0,std430) readonly restrict buffer WeightWords { uint words[]; } ww;
layout(binding=0,std430) readonly restrict buffer WeightHalves { uint16_t halves[]; } wh;
layout(binding=0,std430) readonly restrict buffer WeightHalfVectors { u16vec4 halves4[]; } wh4;
layout(set=2,binding=0,std430) buffer Binding_a_f { float v[]; } a_f;
layout(set=2,binding=0,std430) buffer Binding_a_u { uint v[]; } a_u;
layout(set=2,binding=0,std430) buffer Binding_a_h { uint16_t v[]; } a_h;
layout(set=2,binding=0,std430) buffer Binding_a_v4 { vec4 v[]; } a_v4;
layout(set=2,binding=1,std430) readonly restrict buffer Binding_b_f { float v[]; } b_f;
layout(set=2,binding=1,std430) readonly restrict buffer Binding_b_u { uint v[]; } b_u;
layout(set=2,binding=1,std430) readonly restrict buffer Binding_b_h { uint16_t v[]; } b_h;
layout(set=2,binding=1,std430) readonly restrict buffer Binding_b_v4 { vec4 v[]; } b_v4;
layout(set=2,binding=2,std430) buffer Binding_c_f { float v[]; } c_f;
layout(set=2,binding=2,std430) buffer Binding_c_u { uint v[]; } c_u;
layout(set=2,binding=2,std430) buffer Binding_c_h { uint16_t v[]; } c_h;
layout(set=2,binding=2,std430) buffer Binding_c_v4 { vec4 v[]; } c_v4;
layout(set=2,binding=3,std430) buffer Binding_d_f { float v[]; } d_f;
layout(set=2,binding=3,std430) buffer Binding_d_u { uint v[]; } d_u;
layout(set=2,binding=3,std430) buffer Binding_d_h { uint16_t v[]; } d_h;
layout(set=2,binding=3,std430) buffer Binding_d_v4 { vec4 v[]; } d_v4;
layout(set=2,binding=4,std430) buffer Binding_e_f { float v[]; } e_f;
layout(set=2,binding=4,std430) buffer Binding_e_u { uint v[]; } e_u;
layout(set=2,binding=4,std430) buffer Binding_e_h { uint16_t v[]; } e_h;
layout(set=2,binding=4,std430) buffer Binding_e_v4 { vec4 v[]; } e_v4;
layout(set=2,binding=5,std430) buffer Binding_f_f { float v[]; } f_f;
layout(set=2,binding=5,std430) buffer Binding_f_u { uint v[]; } f_u;
layout(set=2,binding=5,std430) buffer Binding_f_h { uint16_t v[]; } f_h;
layout(set=2,binding=5,std430) buffer Binding_f_v4 { vec4 v[]; } f_v4;
layout(set=2,binding=6,std430) readonly restrict buffer Binding_params_f { float v[]; } params_f;
layout(set=2,binding=6,std430) readonly restrict buffer Binding_params_u { uint v[]; } params_u;
layout(set=2,binding=6,std430) readonly restrict buffer Binding_params_h { uint16_t v[]; } params_h;
layout(set=2,binding=6,std430) readonly restrict buffer Binding_params_v4 { vec4 v[]; } params_v4;
layout(set=2,binding=7,std430) buffer Binding_extra_f { float v[]; } extra_f;
layout(set=2,binding=7,std430) buffer Binding_extra_u { uint v[]; } extra_u;
layout(set=2,binding=7,std430) buffer Binding_extra_h { uint16_t v[]; } extra_h;
layout(set=2,binding=7,std430) buffer Binding_extra_v4 { vec4 v[]; } extra_v4;
layout(push_constant, std430) uniform Push {
    // r2 is a CPU-only flag for deferring the trailing barrier across independent dispatches.
    uint64_t a,b,c,d,e,f,params,extra;
    uint n,k,m,stride,heads,kvheads,dim,cap,window,kind,offset,flags,r0,r1,r2,r3;
} p;
uint position() { return params_u.v[0]; }
uint batch() { return p.r3!=0 ? p.r3 : params_u.v[1]; }
float half_b(uint i) { return unpackHalf2x16(uint(b_h.v[i])).x; }
float half_c(uint i) { return unpackHalf2x16(uint(c_h.v[i])).x; }
// Flip each nibble's sign bit so signed extraction directly yields q-8.
// GCN can then use signed bitfield extracts without eight separate subtractions.
vec4 q4_lo(uint bits) {
    int v=int(bits^0x88888888u);
    return vec4(bitfieldExtract(v,0,4),bitfieldExtract(v,8,4),bitfieldExtract(v,16,4),bitfieldExtract(v,24,4));
}
vec4 q4_hi(uint bits) {
    int v=int(bits^0x88888888u);
    return vec4(bitfieldExtract(v,4,4),bitfieldExtract(v,12,4),bitfieldExtract(v,20,4),bitfieldExtract(v,28,4));
}
float q4(uint64_t ptr,uint block,uint j) {
    uint v=ww.words[p.stride+block*4+(j%16)/4];
    uint q=(v>>(((j%4)*8)+(j/16)*4))&15;
    return float(int(q)-8)*uintBitsToFloat(ww.words[block]);
}
