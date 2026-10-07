// Dense-matrix block access for the fused hyper-connection mixer kernels. FMT 0: Q4_0 with f32 scales
// (quant::repack_q4, blocks from word p.stride), FMT 1: Q8_0 with f16 scales (quant::repack_q8h).
layout(constant_id=0) const uint FMT=0;
// The 8 values of word `part` of a Q4_0 block are elements 4 part..+3 and 16 + 4 part..+3; a Q8_0 block has 8
// words of 4 consecutive values. Returns the scale.
float wscale(uint blk){return FMT==0?uintBitsToFloat(ww.words[blk]):unpackHalf2x16(uint(wh.halves[blk])).x;}
