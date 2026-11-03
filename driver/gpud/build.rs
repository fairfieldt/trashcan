// Compiles Linux's Southern Islands amdgpu files (driver/linux, unmodified) and gpud's C core (driver/shim) into one
// static library. Linux targets only: elsewhere gpud builds without the hardware code (tests use the mock platform).
use std::path::PathBuf;

const IMPORTED: &[&str] = &[
    "amdgpu/si",
    "amdgpu/gmc_v6_0",
    "amdgpu/gfx_v6_0",
    "amdgpu/si_ih",
    "amdgpu/si_dma",
    "amdgpu/amdgpu_ih",
    "amdgpu/amdgpu_wb",
    "amdgpu/atom",
    "amdgpu/amdgpu_atombios",
    "pm/legacy-dpm/si_dpm",
    "pm/legacy-dpm/si_smc",
    "pm/legacy-dpm/legacy_dpm",
];
const CORE: &[&str] = &["lkpi", "device", "mem", "ring", "vm", "info", "pm"];

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() != "linux" {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let amd = root.join("linux/drivers/gpu/drm/amd");
    let mut build = cc::Build::new();
    build
        .std("gnu11")
        .define("_GNU_SOURCE", None)
        // the kernel's own: char is unsigned (Linux 6.2+), pointer signedness and empty printk formats are fine
        .flag("-funsigned-char")
        .flag("-Wno-pointer-sign")
        .flag("-Wno-format-zero-length")
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
        .include(amd.join("amdgpu"))
        .include(amd.join("pm/inc"))
        .include(amd.join("pm/legacy-dpm"))
        .include(amd.join("include"))
        .include(amd.join("include/asic_reg"))
        .include(root.join("linux/include"))
        .include(root.join("linux/include/uapi"));
    for f in IMPORTED {
        build.file(amd.join(format!("{f}.c")));
    }
    for f in CORE {
        build.file(root.join(format!("shim/core/{f}.c")));
    }
    build.compile("gpud_amdgpu");
    println!("cargo:rerun-if-changed={}", root.join("linux").display());
    println!("cargo:rerun-if-changed={}", root.join("shim").display());
}
