//! Dross-only import proof. The GPU writes native shared Memory and the CPU checks
//! that same object, with no CPU frame copy and no host graphics environment.
use anyhow::{Result, anyhow, ensure};
use dross_rt::sys;
use oscar_desktop::canvas::{Painter, Rect};
use oscar_desktop_gpu::{Frame, Renderer};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let bus = args
        .iter()
        .position(|a| a == "--bus")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse())
        .transpose()?;
    // Use a real native Memory object, not anonymous Linux allocation. The native
    // object is the import authority; gpud owns its own backing reference.
    let bytes = 64 * 1024;
    let memory = sys::mem_create(bytes, 0).map_err(|e| anyhow!("native Memory: {e:?}"))?;
    let pointer = sys::vm_map(memory, 0, bytes, 0, 3).map_err(|e| anyhow!("map: {e:?}"))?;
    let reader = sys::vm_map(memory, 0, bytes, 0, 1).map_err(|e| anyhow!("reader: {e:?}"))?;
    let mut original_released = false;
    let result = (|| -> Result<()> {
        let mut gpu = Renderer::new(128, 128, bus, None)?;
        println!("shared graphics device: {}", gpu.device_name);
        unsafe {
            gpu.import_output(pointer as *mut u8, bytes, 128)?;
        }
        println!(
            "native import minimum alignment: {} bytes",
            gpu.imported_alignment
        );
        // The renderer and driver each own independent references. Revoking the
        // caller's mapping/handle must not invalidate either GPU or CPU access.
        sys::vm_unmap(pointer, bytes).map_err(|e| anyhow!("revoke mapping: {e:?}"))?;
        let _ = sys::close(memory);
        original_released = true;
        let mut frame = Frame::new(128, 128)?;
        for iteration in 0..32 {
            frame.reset();
            let color = 0x123400 + iteration;
            frame.fill(frame.bounds(), color);
            frame.fill(Rect::new(23, 19, 60, 47), 0xabcdef);
            let pixels = gpu.render(&frame)?;
            ensure!(
                pixels.as_ptr() as u64 != pointer,
                "renderer borrowed revoked mapping"
            );
            for y in 0..128 {
                for x in 0..128 {
                    let expected = if (23..83).contains(&x) && (19..66).contains(&y) {
                        0xabcdef
                    } else {
                        color
                    };
                    let actual = unsafe { (reader as *const u32).add(y * 128 + x).read_volatile() };
                    ensure!(
                        pixels[y * 128 + x] & 0xffffff == expected,
                        "renderer alias pixel mismatch"
                    );
                    ensure!(
                        actual & 0xffffff == expected,
                        "shared frame {iteration}, ({x},{y}): {actual:08x} != {expected:08x}"
                    );
                }
            }
        }
        println!(
            "ok: 32 native Memory GPU frames, exact pixels, original mapping/handle revoked, no CPU pixel copy"
        );
        drop(gpu);
        Ok(())
    })();
    if !original_released {
        let _ = sys::vm_unmap(pointer, bytes);
        let _ = sys::close(memory);
    }
    let _ = sys::vm_unmap(reader, bytes);
    result
}
