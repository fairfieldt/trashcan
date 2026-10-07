// Dumps reference activations from llama.cpp for validating the trashcan engine.
//
// Runs a prompt through llama.cpp (CPU or Vulkan), and for selected graph tensors writes the LAST token's row as raw
// little-endian f32 to OUT_DIR/<tensor name>.f32, plus OUT_DIR/index.tsv (name, shape of the full tensor, row length).
// Then decodes greedily for N tokens and writes OUT_DIR/greedy.txt (the generated token ids, one per line), and the
// last prompt token's full logits to OUT_DIR/logits.f32.
//
// usage: dump MODEL OUT_DIR cpu|vulkan N_GREEDY TOKEN_ID... (or @FILE with whitespace-separated ids)
//
// Environment: DUMP_DETAIL=0,3 (detail layers, default 0,5; "all" for every layer), DUMP_ALWAYS=base,... (bases
// dumped at every layer, default attn_out,l_out), DUMP_NAMES=name,... (extra exact names), DUMP_CTX (context size,
// default 1024; the prompt runs as one ubatch), DUMP_THREADS, DUMP_FULL=base,... (names or bases written with every
// token's row instead of the last one). A name the graph uses more than once per decode (qwen4exp's two
// hyper-connection mixers per layer) gets "#2", "#3" ... from its second use on. A tensor without a dimension equal
// to the prompt length holds a single token (rows gathered for the output), and is written whole.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct dump_ctx {
    std::string dir;
    std::set<std::string> wanted;  // exact names
    std::set<int> detail_layers;   // layers whose intermediates are all dumped
    bool all_layers = false;
    std::set<std::string> always = { "attn_out", "l_out" };
    std::map<std::string, int> seen; // uses of each name in this decode
    std::set<std::string> full;      // bases written with every token's row
    int64_t n_prompt = 0;
    std::ofstream index;
    bool active = true;            // only dump during the prompt decode
};

static std::set<std::string> split_names(const char * s) {
    std::set<std::string> out;
    std::string cur;
    for (; *s; s++) {
        if (*s == ',') {
            if (!cur.empty()) {
                out.insert(cur);
            }
            cur.clear();
        } else {
            cur += *s;
        }
    }
    if (!cur.empty()) {
        out.insert(cur);
    }
    return out;
}

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
    if (d.always.count(base)) {
        return true;
    }
    return d.all_layers || d.detail_layers.count(layer) > 0;
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
    // [head_dim, n_head, n_tokens]. With one token everything is one row. When the prompt length is known,
    // only a dimension of that length counts, so [n_embd, hc, 1] rows gathered for the output stay whole.
    int64_t n_tok = 1;
    for (int dim = GGML_MAX_DIMS - 1; dim >= 1; dim--) {
        if (d->n_prompt > 1 ? t->ne[dim] == d->n_prompt : t->ne[dim] > 1) {
            n_tok = t->ne[dim];
            break;
        }
    }
    const int uses = ++d->seen[name];
    const std::string file = uses == 1 ? name : name + "#" + std::to_string(uses);
    const size_t dash = name.rfind('-');
    const bool all_rows = d->full.count(name) || (dash != std::string::npos && d->full.count(name.substr(0, dash)));
    const int64_t per_tok = all_rows ? ggml_nelements(t) : ggml_nelements(t) / n_tok;
    std::vector<float> buf(per_tok);
    ggml_backend_tensor_get(t, buf.data(), all_rows ? 0 : (n_tok - 1) * per_tok * sizeof(float), per_tok * sizeof(float));
    std::ofstream f(d->dir + "/" + file + ".f32", std::ios::binary);
    f.write((const char *) buf.data(), buf.size() * sizeof(float));
    d->index << file << "\t" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "\t" << per_tok << "\n";
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
        if (argv[i][0] == '@') {
            // @FILE: whitespace-separated token ids
            std::ifstream f(argv[i] + 1);
            for (llama_token t; f >> t; ) {
                prompt.push_back(t);
            }
            continue;
        }
        prompt.push_back(atoi(argv[i]));
    }
    d.wanted = { "inp_scaled", "result_norm", "result_output" };
    d.detail_layers = { 0, 5 };
    d.n_prompt = (int64_t) prompt.size();
    // DUMP_DETAIL=0,3 selects other detail layers (Gemma's defaults are 0 and 5).
    if (const char * detail = getenv("DUMP_DETAIL")) {
        d.detail_layers.clear();
        if (strcmp(detail, "all") == 0) {
            d.all_layers = true;
        }
        for (const char * s = detail; *s && !d.all_layers; ) {
            d.detail_layers.insert(atoi(s));
            const char * comma = strchr(s, ',');
            s = comma ? comma + 1 : s + strlen(s);
        }
    }
    if (const char * always = getenv("DUMP_ALWAYS")) {
        d.always = split_names(always);
    }
    if (const char * names = getenv("DUMP_NAMES")) {
        for (const auto & n : split_names(names)) {
            d.wanted.insert(n);
        }
    }
    if (const char * full = getenv("DUMP_FULL")) {
        d.full = split_names(full);
    }
    const int n_ctx = getenv("DUMP_CTX") ? atoi(getenv("DUMP_CTX")) : 1024;
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
    // DUMP_NO_REPACK=1 keeps the weights in the file map instead of repacked copies (much less resident memory).
    if (getenv("DUMP_NO_REPACK")) {
        mp.use_extra_bufts = false;
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = std::max<uint32_t>(512, prompt.size());
    cp.n_ubatch = cp.n_batch;
    if (const char * threads = getenv("DUMP_THREADS")) {
        cp.n_threads = cp.n_threads_batch = atoi(threads);
    }
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
