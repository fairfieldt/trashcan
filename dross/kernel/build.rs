use std::fmt::Write;

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg-bins=-T{dir}/link.ld");
    println!("cargo:rerun-if-changed=link.ld");

    // One entry stub per vector: vectors without a CPU error code push a zero, then the vector number.
    let mut s = String::new();
    let with_error = [8, 10, 11, 12, 13, 14, 17, 21, 29, 30];
    for v in 0..256 {
        writeln!(s, ".global isr_{v}\nisr_{v}:").unwrap();
        if !with_error.contains(&v) {
            writeln!(s, "    push 0").unwrap();
        }
        writeln!(s, "    push {v}\n    jmp isr_common").unwrap();
    }
    writeln!(s, ".section .rodata\n.balign 8\n.global ISR_TABLE\nISR_TABLE:").unwrap();
    for v in 0..256 {
        writeln!(s, "    .quad isr_{v}").unwrap();
    }
    writeln!(s, ".text").unwrap();
    std::fs::write(format!("{}/isr.s", std::env::var("OUT_DIR").unwrap()), s).unwrap();
}
