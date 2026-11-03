use std::{env, path::PathBuf, process::Command};
fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for shader in ["quad.vert", "quad.frag"] {
        let source = format!("shaders/{shader}");
        assert!(
            Command::new("glslc")
                .args(["--target-env=vulkan1.1", "-O", &source, "-o"])
                .arg(out.join(format!("{shader}.spv")))
                .status()
                .expect("glslc is required")
                .success()
        );
        println!("cargo:rerun-if-changed={source}");
    }
}
