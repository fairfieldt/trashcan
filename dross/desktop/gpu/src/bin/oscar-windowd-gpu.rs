//! Linux-compat windowd adapter using native OSCR channels and Vulkan/RADV.
//! A supplied scanout capability imports native staging memory and page-flips;
//! otherwise completed graphics copy to the firmware framebuffer.
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_desktop_gpu::{Frame, Renderer};
fn main() {
    let b = Bootstrap::read(Handle(1)).expect("windowd bootstrap");
    let _ = sys::close(Handle(1));
    let width = b
        .value_u64("width")
        .and_then(|v| u32::try_from(v).ok())
        .expect("width");
    let height = b
        .value_u64("height")
        .and_then(|v| u32::try_from(v).ok())
        .expect("height");
    let bus = b.value_u64("gpu-bus").and_then(|v| u32::try_from(v).ok());
    let mut renderer = match Renderer::new(width, height, bus, None) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("windowd graphics: {e:#}");
            std::process::exit(1);
        }
    };
    eprintln!("windowd graphics: {}", renderer.device_name);
    let mut scanout = b.handle("scanout").map(|channel| {
        oscar_desktop_native::scanout::Scanout::acquire(channel).expect("exclusive scanout")
    });
    if let Some(output) = &scanout {
        assert_eq!(output.dimensions(), (width, height));
        let (pointer, bytes, pitch) = output.shared_memory();
        unsafe {
            renderer
                .import_output(pointer, bytes, pitch)
                .expect("native GPU scanout memory import");
        }
        eprintln!(
            "windowd scanout: CRTC{} starts at frame{}",
            output.index, output.frame
        );
    }
    let mut frame = Frame::new(width, height).expect("frame");
    let code = oscar_desktop_native::windowd::run(b, |server, display| {
        frame.reset();
        server.paint(&mut frame).map_err(|_| Error::InvalidArgs)?;
        let pixels = renderer.render(&frame).map_err(|e| {
            eprintln!("windowd graphics: {e:#}");
            Error::NoResources
        })?;
        if let Some(output) = &mut scanout {
            output.present_ready()?;
        } else {
            display.present_pixels(pixels, width);
        }
        Ok(())
    });
    if let Some(output) = &scanout {
        eprintln!(
            "windowd scanout: CRTC{} ends at frame{}, {} completed flips",
            output.index, output.frame, output.presents
        );
    }
    drop(renderer);
    if let Some(output) = &mut scanout {
        if let Err(error) = output.release() {
            eprintln!("windowd firmware restore: {error:?}");
            std::process::exit(1);
        }
        eprintln!("windowd firmware restore: completed");
    }
    drop(scanout);
    std::process::exit(code);
}
