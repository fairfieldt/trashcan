// Dumps reference activations from llama.cpp for validating the trashcan engine.
//
// Runs a prompt through llama.cpp (CPU or Vulkan), and for selected graph tensors writes the LAST token's row as raw
// little-endian f32 to OUT_DIR/<tensor name>.f32, plus OUT_DIR/index.tsv (name, shape of the full tensor, row length).
// Then decodes greedily for N tokens and writes OUT_DIR/greedy.txt (the generated token ids, one per line), and the
// last prompt token's full logits to OUT_DIR/logits.f32.
//
// usage: dump MODEL OUT_DIR cpu|vulkan N_GREEDY TOKEN_ID...

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

struct dump_ctx {
    std::string dir;
    std::set<std::string> wanted;  // exact names
    std::set<int> detail_layers;   // layers whose intermediates are all dumped
    std::ofstream index;
    bool active = true;            // only dump during the prompt decode
};

static bool want(const dump_ctx & d, const std::string & name) {
    if (d.wanted.count(name)) {
        return true;
    }
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos) {
        return false;
    }
    const std::string base = name.substr(0, dash);
    const int layer = atoi(name.c_str() + dash + 1);
    if (base == "attn_out" || base == "l_out") {
        return true;
    }
    return d.detail_layers.count(layer) > 0;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    if (!d->active) {
        return !ask;
    }
    const std::string name = t->name;
    if (ask) {
        return want(*d, name);
    }
    if (!want(*d, name) || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return true;
    }
    // The token dimension is the highest non-singleton dimension above 0: [n_embd, n_tokens] or
    // [head_dim, n_head, n_tokens]. With one token everything is one row.
    int64_t n_tok = 1;
    for (int dim = GGML_MAX_DIMS - 1; dim >= 1; dim--) {
        if (t->ne[dim] > 1) {
            n_tok = t->ne[dim];
            break;
        }
    }
    const int64_t per_tok = ggml_nelements(t) / n_tok;
    std::vector<float> buf(per_tok);
    ggml_backend_tensor_get(t, buf.data(), (n_tok - 1) * per_tok * sizeof(float), per_tok * sizeof(float));
    std::ofstream f(d->dir + "/" + name + ".f32", std::ios::binary);
    f.write((const char *) buf.data(), buf.size() * sizeof(float));
    d->index << name << "\t" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "\t" << per_tok << "\n";
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL OUT_DIR cpu|vulkan N_GREEDY TOKEN_ID...\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    dump_ctx d;
    d.dir = argv[2];
    const bool use_gpu = strcmp(argv[3], "vulkan") == 0;
    const int n_greedy = atoi(argv[4]);
    std::vector<llama_token> prompt;
    for (int i = 5; i < argc; i++) {
        prompt.push_back(atoi(argv[i]));
    }
    d.wanted = { "inp_scaled", "result_norm", "result_output" };
    d.detail_layers = { 0, 5 };
    // DUMP_DETAIL=0,3 selects other detail layers (Gemma's defaults are 0 and 5).
    if (const char * detail = getenv("DUMP_DETAIL")) {
        d.detail_layers.clear();
        for (const char * s = detail; *s; ) {
            d.detail_layers.insert(atoi(s));
            const char * comma = strchr(s, ',');
            s = comma ? comma + 1 : s + strlen(s);
        }
    }
    d.index.open(d.dir + "/index.tsv");

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
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &d;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), prompt.size())) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }
    d.active = false;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    {
        std::ofstream f(d.dir + "/logits.f32", std::ios::binary);
        f.write((const char *) logits, (size_t) n_vocab * sizeof(float));
    }

    std::ofstream g(d.dir + "/greedy.txt");
    llama_token tok = 0;
    for (int i = 0; i < n_greedy; i++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        tok = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (lg[v] > lg[tok]) {
                tok = v;
            }
        }
        g << tok << "\n";
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
            fprintf(stderr, "decode failed at step %d\n", i);
            return 1;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "wrote %s\n", d.dir.c_str());
    return 0;
}
