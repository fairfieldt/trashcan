// Compiles the GLSL compute shaders to SPIR-V with glslc.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for name in ["fma", "copy"] {
        let src = format!("shaders/{name}.comp");
        let status = Command::new("glslc")
            .args(["-O", "--target-env=vulkan1.1", &src, "-o"])
            .arg(out.join(format!("{name}.spv")))
            .status()
            .expect("glslc");
        assert!(status.success(), "glslc {src}");
        println!("cargo:rerun-if-changed={src}");
    }
}
