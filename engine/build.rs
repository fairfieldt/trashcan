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
        "norm_rope",
        "rope",
        "attention",
        "attention_partial",
        "attention_merge",
        "geglu",
        "topk",
        "gemm",
        "visibility",
        "multivec",
        "multivec2",
        "multivec4",
        "matvecq4",
        "matvecq4narrow",
        "merge",
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
