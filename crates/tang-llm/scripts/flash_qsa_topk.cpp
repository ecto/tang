// flash_qsa_topk.cpp: llama.cpp's QSA block selection for the last token of a prompt, to check
// `tang-llm flash-ref --dump`'s `LNN.qsa_selected` against.
//
// Drop-in replacement for llama.cpp's examples/eval-callback/eval-callback.cpp (built as
// `llama-eval-callback`, CPU only):
//
//   QSA_IDS=ids.txt QSA_OUT=dir llama-eval-callback -m model.gguf -c 9216 -t 16
//
// reads whitespace-separated token ids from $QSA_IDS, decodes them in one batch, and for every QSA
// layer writes $QSA_OUT/topk-<layer>.txt and score-<layer>.txt: the last ubatch's `indexer_top_k`
// (selected pool = block indices, unordered, per token) and `indexer_score` (per pool, per token).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::string g_out;

// The whole tensor as text: a header with ne[0..4], then every element in ggml order (ne0 fastest).
// The names come from cb() in qwen4exp.cpp and may land on a reshaped view, so the shape is
// recorded rather than assumed; flash_qsa_compare.py takes the last token's slice.
static void write_tensor(const ggml_tensor * t, const char * kind, int il) {
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), buf.data(), 0, buf.size());
    std::string path = g_out + "/" + kind + "-" + std::to_string(il) + ".txt";
    FILE * f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "# %s layer %d ne %lld %lld %lld %lld type %d\n", kind, il, (long long) t->ne[0],
                 (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], (int) t->type);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const uint8_t * p = buf.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (t->type == GGML_TYPE_I32) {
            int32_t v; std::memcpy(&v, p, 4); std::fprintf(f, "%d\n", v);
        } else {
            float v; std::memcpy(&v, p, 4); std::fprintf(f, "%.9g\n", v);
        }
    }
    std::fclose(f);
}

static bool cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const bool topk = std::strncmp(t->name, "indexer_top_k-", 14) == 0;
    const bool score = std::strncmp(t->name, "indexer_score-", 14) == 0;
    if (ask) return topk || score;
    if (topk || score) {
        write_tensor(t, topk ? "topk" : "score", std::atoi(t->name + 14));
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    const char * ids_path = std::getenv("QSA_IDS");
    const char * out = std::getenv("QSA_OUT");
    if (!ids_path || !out) { LOG_ERR("set QSA_IDS and QSA_OUT\n"); return 1; }
    g_out = out;
    std::vector<llama_token> tokens;
    {
        std::ifstream f(ids_path);
        long long v;
        while (f >> v) tokens.push_back((llama_token) v);
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = cb;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;
    auto llama_init = common_init_from_params(params);
    auto * ctx = llama_init->context();
    if (!ctx) { LOG_ERR("failed to init\n"); return 1; }
    LOG_INF("decoding %zu tokens\n", tokens.size());
    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { LOG_ERR("decode failed\n"); return 1; }
    llama_backend_free();
    return 0;
}
