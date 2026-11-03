//! Offscreen graphics correctness on the exact renderer used by windowd.
//! A successful host run proves Vulkan drawing, not gpud or physical scanout.
use anyhow::{Result, ensure};
use oscar_desktop::{
    canvas::{Canvas, Painter, Rect},
    protocol::{Input, Message, Packet},
    server::Server,
};
use oscar_desktop_gpu::{Frame, Renderer};
fn compare(actual: &[u32], expected: &[u32], tolerance: i32) -> Result<()> {
    ensure!(actual.len() == expected.len(), "pixel count mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        for shift in [0, 8, 16] {
            ensure!(
                (((a >> shift) & 255) as i32 - ((e >> shift) & 255) as i32).abs() <= tolerance,
                "pixel {i}: {a:08x} != {e:08x} (tolerance {tolerance})"
            );
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let library = value("--library").map(std::path::Path::new);
    let bus = value("--bus").map(|s| s.parse()).transpose()?;
    let (width, height) = (640, 480);
    let mut gpu = Renderer::new(width, height, bus, library)?;
    println!("graphics device: {}", gpu.device_name);
    let mut frame = Frame::new(width, height)?;
    let mut expected = Canvas::new(width, height).unwrap();
    // Exact solid/texture pixels exercise rasterization, fragment storage reads,
    // clipping and transfer readback; this is not a compute dispatch.
    let mut texture = Canvas::new(37, 23).unwrap();
    for (i, p) in texture.pixels_mut().iter_mut().enumerate() {
        *p = ((i as u32 * 193) & 0xffffff) | 0x102030;
    }
    for target in [
        &mut frame as &mut dyn Painter,
        &mut expected as &mut dyn Painter,
    ] {
        target.desktop(0x5b6469);
        target.fill(Rect::new(-8, 7, 120, 90), 0x123456);
        target.fill(Rect::new(630, 460, 300, 300), 0xff0055);
        target.blit(&texture, 120, 33);
        target.blit(&texture, -11, 400);
    }
    compare(gpu.render(&frame)?, expected.pixels(), 0)?;
    println!("ok: exact solid, clipped texture and desktop pattern pixels");
    let mut server = Server::new(width, height).unwrap();
    for (client, title) in [(1, "Terminal"), (2, "Doom")] {
        let replies = server.request(
            client,
            Packet::new(
                1,
                0,
                Message::Create {
                    width: 240,
                    height: 150,
                    title: title.into(),
                },
            ),
        );
        let id = replies
            .iter()
            .find(|d| matches!(d.packet.message, Message::Created { .. }))
            .unwrap()
            .packet
            .window;
        let pixels = (0..240 * 150)
            .map(|i| {
                if (i / 240 + i % 240) % 17 < 8 {
                    0x2468ac
                } else {
                    0xe0d090
                }
            })
            .collect();
        server.request(
            client,
            Packet::new(2, id, Message::Upload { offset: 0, pixels }),
        );
        server.request(client, Packet::new(2, id, Message::Present));
    }
    for action in [
        None,
        Some(Input::Pointer {
            x: 29,
            y: 49,
            down: true,
        }),
        Some(Input::Pointer {
            x: 180,
            y: 100,
            down: true,
        }),
        Some(Input::Pointer {
            x: 180,
            y: 100,
            down: false,
        }),
    ] {
        if let Some(input) = action {
            server.input(input);
        }
        let reference = server.render().pixels().to_vec();
        frame.reset();
        server.paint(&mut frame).unwrap();
        // UNORM hardware alpha blending rounds, while the integer CPU glyph
        // renderer truncates. At most one code value per channel is permitted.
        compare(gpu.render(&frame)?, &reference, 1)?;
    }
    println!(
        "ok: overlapping native windows, decorations, focus, drag and glyphs (<=1 glyph rounding)"
    );
    server.disconnect(1);
    server.disconnect(2);
    frame.reset();
    server.paint(&mut frame).unwrap();
    compare(gpu.render(&frame)?, server.render().pixels(), 0)?;
    println!("ok: disconnect clears window content");
    Ok(())
}
