// MTP draft reference from llama.cpp (a build with draft-mtp, e.g. b8e9ce8), for validating `trashcan qwen4 mtp ref`.
//
// Runs a prompt through the target model on the CPU (keeping the hc-wide residual of every token), then the MTP
// block of an MTP-only GGUF over the prompt as llama.cpp's draft-mtp driver does (token p paired with the target's
// residual row p - 1, zeros at 0), and writes the block's last-row intermediates and draft logits to
// OUT_DIR/<name>.f32 with OUT_DIR/index.tsv (the format of dump.cpp). Then CHAIN draft steps: the target's greedy
// next token at position n with residual row n - 1, then each draft with the block's own output
// (draft{i}_logits.f32, drafts.json).
//
// usage: mtp MODEL MTP_GGUF OUT_DIR CHAIN TOKEN_ID... (or @FILE with whitespace-separated ids)
// Environment: DUMP_THREADS, DUMP_CTX (default 1024).

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct cap_ctx {
    std::string dir;
    std::map<std::string, int> seen;
    std::ofstream index;
    bool active = true;
};

// Every f32 tensor of the MTP graph: the last row along the highest non-singleton dimension above 0.
static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (cap_ctx *) user_data;
    if (ask) {
        return d->active;
    }
    if (!d->active || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return true;
    }
    int64_t n_tok = 1;
    for (int dim = GGML_MAX_DIMS - 1; dim >= 1; dim--) {
        if (t->ne[dim] > 1) {
            n_tok = t->ne[dim];
            break;
        }
    }
    const std::string name = t->name;
    // [n_embd, hc, n_tokens] residuals keep all streams of the last token
    if (t->ne[2] > 1 && t->ne[1] > 1 && t->ne[3] == 1) {
        n_tok = t->ne[2];
    }
    const int uses = ++d->seen[name];
    const std::string file = uses == 1 ? name : name + "#" + std::to_string(uses);
    const int64_t per_tok = ggml_nelements(t) / n_tok;
    std::vector<float> buf(per_tok);
    ggml_backend_tensor_get(t, buf.data(), (n_tok - 1) * per_tok * sizeof(float), per_tok * sizeof(float));
    std::ofstream f(d->dir + "/" + file + ".f32", std::ios::binary);
    f.write((const char *) buf.data(), buf.size() * sizeof(float));
    d->index << file << "\t" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "\t" << per_tok << "\n";
    return true;
}

static llama_token argmax(const float * l, int n) {
    llama_token best = 0;
    for (int v = 1; v < n; v++) {
        if (l[v] > l[best]) {
            best = v;
        }
    }
    return best;
}

static void write_f32(const std::string & path, const float * v, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) v, n * sizeof(float));
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL MTP_GGUF OUT_DIR CHAIN TOKEN_ID...\n", argv[0]);
        return 1;
    }
    cap_ctx d;
    d.dir = argv[3];
    const int chain = atoi(argv[4]);
    std::vector<llama_token> prompt;
    for (int i = 5; i < argc; i++) {
        if (argv[i][0] == '@') {
            std::ifstream f(argv[i] + 1);
            for (llama_token t; f >> t; ) {
                prompt.push_back(t);
            }
            continue;
        }
        prompt.push_back(atoi(argv[i]));
    }
    const int n = (int) prompt.size();
    const int n_ctx = getenv("DUMP_CTX") ? atoi(getenv("DUMP_CTX")) : 1024;
    d.index.open(d.dir + "/index.tsv");

    llama_backend_init();
    ggml_backend_load_all();

    static ggml_backend_dev_t no_devices[] = { nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.devices = no_devices;
    // weights stay in the file map (no repacked copies)
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    llama_model_params mpd = mp;
    mpd.load_mtp = true;
    mpd.use_extra_bufts = getenv("MTP_REPACK") != nullptr;
    llama_model * draft = llama_model_load_from_file(argv[2], mpd);
    if (!model || !draft) {
        fprintf(stderr, "failed to load the models\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = std::max<uint32_t>(512, n);
    cp.n_ubatch = cp.n_batch;
    if (const char * threads = getenv("DUMP_THREADS")) {
        cp.n_threads = cp.n_threads_batch = atoi(threads);
    }
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the target context\n");
        return 1;
    }
    llama_set_embeddings_nextn(ctx, true, false);
    fprintf(stderr, "mtp-tool: target decode\n");
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), n)) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int w = llama_model_n_embd_out(model);
    const llama_token next = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
    std::vector<float> h(llama_get_embeddings_nextn(ctx), llama_get_embeddings_nextn(ctx) + (size_t) n * w);

    llama_context_params cpd = cp;
    cpd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    if (getenv("MTP_FA")) {
        cpd.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    cpd.ctx_other = ctx;
    if (!getenv("MTP_NO_CAPTURE")) {
        cpd.cb_eval = cb_eval;
        cpd.cb_eval_user_data = &d;
    }
    llama_context * dctx = llama_init_from_model(draft, cpd);
    if (!dctx) {
        fprintf(stderr, "failed to create the MTP context\n");
        return 1;
    }
    llama_set_embeddings_nextn(dctx, true, true);

    // As llama.cpp's draft-mtp driver: the prompt without outputs, then the last prompt token alone with its logits.
    std::vector<float> zeros(w, 0.0f);
    llama_batch_ext * b = llama_batch_ext_init(dctx);
    auto add = [&](int p, bool output) {
        const int32_t idx = llama_batch_ext_add_token(b, 0, prompt[p]);
        const llama_pos pos[4] = { p, 0, 0, 0 }; // as common_batch: up to 4 M-RoPE positions
        llama_batch_ext_set_pos(b, idx, pos);
        const float * row = p == 0 ? zeros.data() : h.data() + (size_t) (p - 1) * w;
        llama_batch_ext_set_embd_token(b, idx, { row, 1, (size_t) w });
        llama_batch_ext_set_output_logits(b, idx, output);
    };
    if (n > 1) {
        for (int p = 0; p + 1 < n; p++) {
            add(p, false);
        }
        fprintf(stderr, "mtp-tool: MTP prompt decode\n");
        if (llama_process(dctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) {
            fprintf(stderr, "MTP prompt decode failed\n");
            return 1;
        }
        llama_batch_ext_clear(b);
    }
    add(n - 1, true);
    fprintf(stderr, "mtp-tool: MTP last prompt token\n");
    if (llama_process(dctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) {
        fprintf(stderr, "MTP decode failed\n");
        return 1;
    }
    d.active = false;
    const llama_token prompt_draft = argmax(llama_get_logits_ith(dctx, -1), n_vocab);
    write_f32(d.dir + "/result_output.f32", llama_get_logits_ith(dctx, -1), n_vocab);

    std::vector<float> hh(h.end() - w, h.end());
    llama_token tok = next;
    std::vector<llama_token> drafts;
    for (int i = 0; i < chain; i++) {
        llama_batch_ext_clear(b);
        const int32_t idx = llama_batch_ext_add_token(b, 0, tok);
        const llama_pos pos[4] = { n + i, 0, 0, 0 };
        llama_batch_ext_set_pos(b, idx, pos);
        llama_batch_ext_set_embd_token(b, idx, { hh.data(), 1, (size_t) w });
        llama_batch_ext_set_output_logits(b, idx, true);
        if (llama_process(dctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) {
            fprintf(stderr, "draft step %d failed\n", i);
            return 1;
        }
        const float * l = llama_get_logits_ith(dctx, -1);
        write_f32(d.dir + "/draft" + std::to_string(i) + "_logits.f32", l, n_vocab);
        tok = argmax(l, n_vocab);
        drafts.push_back(tok);
        const float * out = llama_get_embeddings_nextn_ith(dctx, -1);
        hh.assign(out, out + w);
    }
    std::ofstream j(d.dir + "/drafts.json");
    j << "{\"prompt_next\": " << next << ", \"prompt_draft\": " << prompt_draft << ", \"chain\": [";
    for (size_t i = 0; i < drafts.size(); i++) {
        j << (i ? ", " : "") << drafts[i];
    }
    j << "]}\n";
    printf("next %d, draft over the prompt %d, chained drafts", next, prompt_draft);
    for (auto t : drafts) {
        printf(" %d", t);
    }
    printf("\n");

    llama_batch_ext_free(b);
    llama_free(dctx);
    llama_free(ctx);
    llama_model_free(draft);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
