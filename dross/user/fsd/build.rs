use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let freestanding = target == "x86_64-unknown-none";
    let mut objects = Vec::new();
    let mut sources: Vec<_> = fs::read_dir("vendor/lwext4/src")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|s| s == "c")
                && !matches!(
                    p.file_stem().unwrap().to_str().unwrap(),
                    "ext4_mkfs" | "ext4_mbr" | "ext4"
                )
        })
        .collect();
    sources.sort();
    sources.push("c/adapter.c".into());
    if freestanding {
        sources.push("c/libc.c".into());
    }
    for src in sources {
        let object = out.join(src.file_stem().unwrap()).with_extension("o");
        let mut cmd = Command::new(if freestanding { "zig" } else { "cc" });
        if freestanding {
            cmd.args([
                "cc",
                "-target",
                "x86_64-linux-musl",
                "-ffreestanding",
                "-fno-stack-protector",
                "-mno-red-zone",
                "-fno-pic",
            ]);
        }
        cmd.args([
            "-O2",
            "-std=gnu11",
            "-DCONFIG_USE_DEFAULT_CFG=1",
            "-DCONFIG_HAVE_OWN_ERRNO=1",
            "-DCONFIG_DEBUG_PRINTF=0",
            "-DCONFIG_DEBUG_ASSERT=0",
            "-DCONFIG_BLOCK_DEV_CACHE_SIZE=32",
            "-DCONFIG_MAX_TRUNCATE_SIZE=1048576",
            "-Ivendor/lwext4/include",
            "-Ic/include",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&object);
        assert!(
            cmd.status()
                .expect("C compiler (Zig required for native fsd)")
                .success(),
            "compile {}",
            src.display()
        );
        objects.push(object);
    }
    let archive = out.join("liblwext4.a");
    if archive.exists() {
        fs::remove_file(&archive).unwrap();
    }
    let mut ar = Command::new(if freestanding { "zig" } else { "ar" });
    if freestanding {
        ar.arg("ar");
    }
    ar.arg("crs").arg(&archive).args(&objects);
    assert!(ar.status().unwrap().success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=lwext4");
    println!("cargo:rerun-if-changed=vendor/lwext4");
    println!("cargo:rerun-if-changed=c");
}
