// Writes a copy of a GGUF whose quantized tensors are dequantized with ggml's own routines, so that llama.cpp's CPU
// backend multiplies by numerically the same weights without rounding the activations to 8 bits: Q8_0 (and any other
// quantized type) becomes F32; NVFP4 becomes F16, which holds every E2M1 value times its E4M3 scale exactly (the
// per-expert f32 scales are separate tensors). The conversion to F16 is checked to be exact. Unquantized tensors and
// all metadata are copied as they are.
//
// usage: dequant IN.gguf OUT.gguf

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s IN.gguf OUT.gguf\n", argv[0]);
        return 1;
    }
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ &meta };
    gguf_context * in = gguf_init_from_file(argv[1], ip);
    if (!in) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    const int64_t n = gguf_get_n_tensors(in);
    std::vector<ggml_type> to(n);
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        gguf_add_tensor(out, t);
        to[i] = t->type == GGML_TYPE_NVFP4 ? GGML_TYPE_F16 : ggml_is_quantized(t->type) ? GGML_TYPE_F32 : t->type;
        if (to[i] != t->type) {
            gguf_set_tensor_type(out, name, to[i]);
        }
    }
    std::vector<uint8_t> head(gguf_get_meta_size(out));
    gguf_get_meta_data(out, head.data());
    std::ifstream src(argv[1], std::ios::binary);
    std::ofstream dst(argv[2], std::ios::binary);
    dst.write((const char *) head.data(), head.size());
    const size_t in_base = gguf_get_data_offset(in), out_base = head.size();
    std::vector<uint8_t> raw, conv;
    std::vector<float> f32;
    size_t written = 0;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const int64_t ne = ggml_nelements(t);
        raw.resize(gguf_get_tensor_size(in, i));
        src.seekg(in_base + gguf_get_tensor_offset(in, i));
        src.read((char *) raw.data(), raw.size());
        const uint8_t * data = raw.data();
        size_t size = raw.size();
        if (to[i] != t->type) {
            f32.resize(ne);
            ggml_get_type_traits(t->type)->to_float(raw.data(), f32.data(), ne);
            if (to[i] == GGML_TYPE_F16) {
                conv.resize(ne * sizeof(ggml_fp16_t));
                ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) conv.data(), ne);
                std::vector<float> back(ne);
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) conv.data(), back.data(), ne);
                if (memcmp(back.data(), f32.data(), ne * sizeof(float)) != 0) {
                    fprintf(stderr, "%s: F16 does not hold the dequantized values exactly\n", name);
                    return 1;
                }
            } else {
                conv.assign((const uint8_t *) f32.data(), (const uint8_t *) (f32.data() + ne));
            }
            data = conv.data();
            size = conv.size();
        }
        const size_t at = out_base + gguf_get_tensor_offset(out, i);
        if (at < out_base + written) {
            fprintf(stderr, "%s: overlapping offsets\n", name);
            return 1;
        }
        std::vector<char> pad(out_base + written > at ? 0 : at - out_base - written, 0);
        dst.write(pad.data(), pad.size());
        dst.write((const char *) data, size);
        written = at - out_base + size;
        if (to[i] != t->type) {
            fprintf(stderr, "%s: %s -> %s\n", name, ggml_type_name(t->type), ggml_type_name(to[i]));
        }
    }
    if (!dst) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    fprintf(stderr, "wrote %s: %zu bytes of tensor data\n", argv[2], written);
    gguf_free(in);
    gguf_free(out);
    return 0;
}
