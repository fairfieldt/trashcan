// Writes llama.cpp's full logits for every position of fixed-size chunks of a token stream, for KL-divergence
// comparisons with the trashcan engine (scripts/inkling-kld.py) or between two llama.cpp runs.
//
// The token ids are split into chunks of CTX tokens (the last partial chunk is dropped), as llama-perplexity does.
// Every chosen chunk starts from an empty cache, runs in batches of LOGITS_BATCH tokens (default 512) and writes
// OUT_DIR/chunk-NNN.f32: CTX rows of n_vocab little-endian f32 logits, row i being the prediction after token i.
//
// usage: logits MODEL IDS_FILE OUT_DIR CTX CHUNKS cpu|gpu
//   CHUNKS: "all" or a comma-separated list of chunk indices.
// Environment: LOGITS_THREADS (CPU threads), LOGITS_BATCH, LOGITS_KV=f16|f32 (K/V cache type, default f16, as
// llama.cpp's default), LOGITS_NO_REPACK=1 (keep weights in the file map).
// cpu hides every accelerator from llama.cpp (also set CUDA_VISIBLE_DEVICES= to keep CUDA from initialising).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL IDS_FILE OUT_DIR CTX CHUNKS cpu|gpu\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string dir = argv[3];
    const int n_ctx = atoi(argv[4]);
    const bool use_gpu = strcmp(argv[6], "gpu") == 0;
    std::vector<llama_token> ids;
    {
        std::ifstream f(argv[2]);
        for (llama_token t; f >> t; ) {
            ids.push_back(t);
        }
    }
    const int n_chunk_max = (int) ids.size() / n_ctx;
    std::vector<int> chunks;
    if (strcmp(argv[5], "all") == 0) {
        for (int c = 0; c < n_chunk_max; c++) {
            chunks.push_back(c);
        }
    } else {
        for (const char * s = argv[5]; *s; ) {
            chunks.push_back(atoi(s));
            const char * comma = strchr(s, ',');
            s = comma ? comma + 1 : s + strlen(s);
        }
    }
    for (int c : chunks) {
        if (c < 0 || c >= n_chunk_max) {
            fprintf(stderr, "chunk %d out of range (%zu tokens make %d chunks of %d)\n", c, ids.size(), n_chunk_max, n_ctx);
            return 1;
        }
    }
    const int n_batch = getenv("LOGITS_BATCH") ? atoi(getenv("LOGITS_BATCH")) : 512;
    const bool kv_f32 = getenv("LOGITS_KV") && strcmp(getenv("LOGITS_KV"), "f32") == 0;

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    static ggml_backend_dev_t no_devices[] = { nullptr };
    if (use_gpu) {
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    } else {
        mp.n_gpu_layers = 0;
        mp.devices = no_devices;
    }
    if (getenv("LOGITS_NO_REPACK")) {
        mp.use_extra_bufts = false;
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    if (const char * threads = getenv("LOGITS_THREADS")) {
        cp.n_threads = cp.n_threads_batch = atoi(threads);
    }
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (kv_f32) {
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    {
        std::ofstream meta(dir + "/meta.tsv");
        meta << "model\t" << model_path << "\nids\t" << argv[2] << "\nctx\t" << n_ctx << "\nvocab\t" << n_vocab
             << "\nbatch\t" << n_batch << "\nbackend\t" << argv[6] << "\nkv\t" << (kv_f32 ? "f32" : "f16") << "\n";
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<float> out((size_t) n_ctx * n_vocab);
    for (int c : chunks) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int b0 = 0; b0 < n_ctx; b0 += n_batch) {
            const int n = std::min(n_batch, n_ctx - b0);
            batch.n_tokens = n;
            for (int k = 0; k < n; k++) {
                batch.token[k] = ids[(size_t) c * n_ctx + b0 + k];
                batch.pos[k] = b0 + k;
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed in chunk %d at %d\n", c, b0);
                return 1;
            }
            for (int k = 0; k < n; k++) {
                memcpy(out.data() + (size_t) (b0 + k) * n_vocab, llama_get_logits_ith(ctx, k), (size_t) n_vocab * sizeof(float));
            }
        }
        char name[64];
        snprintf(name, sizeof name, "/chunk-%03d.f32", c);
        std::ofstream f(dir + name, std::ios::binary);
        f.write((const char *) out.data(), out.size() * sizeof(float));
        if (!f) {
            fprintf(stderr, "failed writing %s%s\n", dir.c_str(), name);
            return 1;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "chunk %d: %d tokens in %.1f s\n", c, n_ctx, s);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
