use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let available = Command::new("glslc").arg("--version").output().is_ok();
    if !available {
        println!("cargo:warning=glslc missing; GPU execution needs a build on incite");
    }
    for name in [
        "matvec",
        "headvec",
        "norm",
        "norm_fused",
        "norm_rope",
        "rope",
        "attention",
        "attn_scores",
        "attn_softmax",
        "attn_values",
        "attention_partial",
        "attention_partial_heads2",
        "attention_partial_heads8",
        "attention_merge",
        "attention_merge_u",
        "attention_merge_reference",
        "geglu",
        "topk",
        "gemm",
        "gemm2",
        "visibility",
        "multivec",
        "multivec2",
        "multivec4",
        "multivec4x2",
        "multivec4x3",
        "multivec4_reference",
        "matvecq4",
        "matvecq4narrow",
        "attention_partial_p2",
        "matvecq4_r2",
        "matvecq4_r2u2",
        "merge",
        "qwen_add_norm",
        "qwen_add_norm_wide",
        "qwen_matf32",
        "qwen_conv",
        "qwen_gdn",
        "qwen_gated_norm",
        "qwen_qk",
        "qwen_gate",
        "qwen_swiglu",
        "qwen_q4c2",
        "qwen_q4c3",
        "qwen_q4c4",
        "qwen_topk",
        "qwen_attn_scores",
        "qwen_attn_softmax",
        "qwen_attn_values",
        // Qwen3.8-Flash-Next (qwen4exp)
        "qwen4_hc_norm",
        "qwen4_hc_mix",
        "qwen4_act",
        "qwen4_gated_norm",
        "qwen4_gdn",
        "qwen4_topk",
        "qwen4_moe_gu",
        "qwen4_moe_down",
        "qwen4_moe_tiles",
        "qwen4_moe_gu_tiles",
        "qwen4_moe_down_tiles",
        "qwen4_moe_sum",
        "qwen4_moe_down_w",
        "qwen4_ple_gate",
        "qwen4_ple_sums",
        "qwen4_combine",
        "qwen4_matf32",
        "qwen4_ple_conv",
        "qwen4_idx_store",
        "qwen4_idx_pool",
        "qwen4_idx_select",
        "qwen4_idx_score",
        "qwen4_attention_sparse",
        "qwen4_attention_sparse_full",
        "qwen4_attention_sparse_q8",
        "qwen4_attention_sparse_full_q8",
        "qwen4_attention_sparse_k16v8",
        "qwen4_attention_sparse_full_k16v8",
        "qwen4_kv_store",
        "qwen4_q8_mvw4_r4",
        "qwen4_q8_mvw4_r2",
        "qwen4_q8_gemm",
        "qwen4_mix_down",
        "qwen4_mix_up",
        "qwen4_mix_act",
        "qwen4_mix_down_sk",
        "qwen4_mix_down_sk4",
        // Cross-machine all-reduce benchmark
        "allreduce_partial",
        "allreduce_check",
        "allreduce_work",
    ] {
        let dest = out.join(format!("{name}.spv"));
        if available {
            let status = Command::new("glslc")
                .args([
                    "--target-env=vulkan1.2",
                    "-O",
                    "-Ishaders",
                    &format!("shaders/{name}.comp"),
                    "-o",
                ])
                .arg(&dest)
                .status()
                .unwrap();
            assert!(status.success(), "shader {name} failed");
        } else {
            fs::write(dest, []).unwrap();
        }
    }
}
