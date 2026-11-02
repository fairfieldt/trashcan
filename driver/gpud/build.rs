// Compiles Linux's Southern Islands amdgpu files (driver/linux, unmodified) and gpud's C core (driver/shim) into one
// static library. Linux targets only: elsewhere gpud builds without the hardware code (tests use the mock platform).
use std::path::PathBuf;

const IMPORTED: &[&str] = &["si", "gmc_v6_0", "gfx_v6_0", "si_ih", "si_dma", "amdgpu_ih", "amdgpu_wb"];
const CORE: &[&str] = &["lkpi", "device", "mem", "ring", "vm", "info"];

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() != "linux" {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let amdgpu = root.join("linux/drivers/gpu/drm/amd/amdgpu");
    let mut build = cc::Build::new();
    build
        .std("gnu11")
        .define("_GNU_SOURCE", None)
        .flag("-fno-strict-aliasing")
        .flag("-fno-common")
        .flag("-Wno-unused-function")
        .flag("-Wno-unused-variable")
        .flag("-Wno-unused-but-set-variable")
        .flag("-Wno-missing-braces")
        .flag("-Wno-missing-field-initializers")
        .flag("-Wno-sign-compare")
        .flag("-Wno-unused-parameter")
        .include(root.join("shim/include"))
        .include(&amdgpu)
        .include(root.join("linux/drivers/gpu/drm/amd/include"))
        .include(root.join("linux/drivers/gpu/drm/amd/include/asic_reg"))
        .include(root.join("linux/include"))
        .include(root.join("linux/include/uapi"));
    for f in IMPORTED {
        build.file(amdgpu.join(format!("{f}.c")));
    }
    for f in CORE {
        build.file(root.join(format!("shim/core/{f}.c")));
    }
    build.compile("gpud_amdgpu");
    println!("cargo:rerun-if-changed={}", root.join("linux").display());
    println!("cargo:rerun-if-changed={}", root.join("shim").display());
}
