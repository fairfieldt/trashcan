//! GPU application drawing presented through the native Oscar shared-window API.
//! The transfer destination imports native shared Memory; no CPU pixel copy.
use anyhow::{Result, anyhow, ensure};
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_desktop::{
    canvas::{Painter, Rect},
    protocol::Input,
};
use oscar_desktop_gpu::{Frame, Renderer};
use oscar_desktop_native::window::{Event, Window};
fn main() -> Result<()> {
    let b = Bootstrap::read(Handle(1)).map_err(|e| anyhow!("bootstrap: {e:?}"))?;
    let _ = sys::close(Handle(1));
    let mut window = Window::open(
        b.handle("window")
            .ok_or_else(|| anyhow!("window capability missing"))?,
        480,
        320,
        "Vulkan",
    )
    .map_err(|e| anyhow!("window: {e:?}"))?;
    let mut gpu = Renderer::new(480, 320, b.value_u64("gpu-bus").map(|b| b as u32), None)?;
    let (pointer, bytes) = window.shared_memory();
    unsafe {
        gpu.import_output(pointer, bytes, 480)?;
    }
    let mut frame = Frame::new(480, 320)?;
    let observer = b.handle("check");
    let mut frames = 0u64;
    let mut acknowledged = 0u64;
    let mut published_position = (80i32, 70i32);
    let mut x = 80;
    let mut y = 70;
    loop {
        for _ in 0..32 {
            match window.poll(0) {
                Ok(Some(Event::Close)) => return Ok(()),
                Ok(Some(Event::Input(Input::Pointer { x: px, y: py, .. }))) => {
                    x = px;
                    y = py;
                }
                Ok(_) => {}
                Err(Error::ShouldWait | Error::TimedOut) => break,
                Err(e) => return Err(anyhow!("window: {e:?}")),
            }
        }
        if window.can_present() {
            if frames > acknowledged {
                if let Some(observer) = observer {
                    let mut state = Vec::new();
                    state.extend_from_slice(&frames.to_le_bytes());
                    state.extend_from_slice(&published_position.0.to_le_bytes());
                    state.extend_from_slice(&published_position.1.to_le_bytes());
                    state.extend_from_slice(&0xf3c54du32.to_le_bytes());
                    let _ = sys::channel_write(observer, &state, &[]);
                }
                acknowledged = frames;
            }
            frame.reset();
            frame.desktop(0x202526);
            frame.fill(Rect::new(40, 40, 220, 170), 0x287caf);
            frame.fill(Rect::new(x - 24, y - 24, 96, 96), 0xf3c54d);
            let pixels = gpu.render(&frame)?;
            if observer.is_some() && (0..480).contains(&x) && (0..320).contains(&y) {
                ensure!(
                    pixels[y as usize * 480 + x as usize] & 0xffffff == 0xf3c54d,
                    "native GPU output pixel mismatch"
                );
            }
            window
                .present_ready()
                .map_err(|e| anyhow!("present: {e:?}"))?;
            frames += 1;
            published_position = (x, y);
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
