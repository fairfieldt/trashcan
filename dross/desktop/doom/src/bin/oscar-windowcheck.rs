//! Dross-only native window integration exercise: starts real DoomGeneric and
//! native windowd, checks framebuffer progress, input, occlusion and shutdown.
use dross_rt::{
    Error, Handle,
    bootstrap::{Bootstrap, Builder},
    sys,
};
use oscar_desktop::{
    canvas::Canvas,
    protocol::{Input, Message, Packet},
};
use oscar_desktop_native::window::{Event, Window};
fn packet(ch: Handle, input: Input) {
    sys::channel_write(
        ch,
        &Packet::new(0, 0, Message::Input(input)).encode().unwrap(),
        &[],
    )
    .unwrap();
}
fn pause(ms: u64) {
    std::thread::sleep(std::time::Duration::from_millis(ms));
}
fn main() {
    let b = Bootstrap::read(Handle(1)).unwrap();
    let _ = sys::close(Handle(1));
    let (width, height) = (
        b.value_u64("width").unwrap_or(800),
        b.value_u64("height").unwrap_or(600),
    );
    let size = (width * height * 4).next_multiple_of(4096);
    let framebuffer = b
        .handle("display")
        .unwrap_or_else(|| sys::mem_create(size, 0).unwrap());
    let size = sys::mem_size(framebuffer).unwrap();
    let stride = b.value_u64("stride").unwrap_or(width);
    let reader =
        sys::vm_map(framebuffer, 0, size, 0, 1).unwrap() + b.value_u64("offset").unwrap_or(0);
    let (input, seat) = sys::channel_create().unwrap();
    let (control, lifetime) = sys::channel_create().unwrap();
    let (server, client) = sys::channel_create().unwrap();
    let (testserver, testclient) = sys::channel_create().unwrap();
    let boot = Builder::default()
        .handle("display", sys::dup(framebuffer, 255).unwrap())
        .handle("input", seat)
        .handle("control", lifetime)
        .handle("client/doom", server)
        .handle("client/check", testserver)
        .value("width", &width.to_string())
        .value("height", &height.to_string())
        .value("stride", &stride.to_string())
        .value("format", b.value("format").unwrap_or("bgr"))
        .value("offset", b.value("offset").unwrap_or("0"))
        .value("gpu-bus", b.value("gpu-bus").unwrap_or("6"));
    let windowd = if let Some(image) = b.handle("windowd-gpu") {
        let mut environment: Vec<String> = [
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "VK_DRIVER_FILES",
            "GPUD_SOCKET",
        ]
        .iter()
        .filter_map(|key| b.value(key).map(|value| format!("{key}={value}")))
        .collect();
        environment.push("DROSS_BOOTSTRAP=1".into());
        let env: Vec<&str> = environment.iter().map(String::as_str).collect();
        let bootstrap = boot.send().unwrap();
        sys::spawn_linux(image, &["oscar-windowd-gpu"], &env, Some(bootstrap)).unwrap()
    } else {
        dross_rt::bootstrap::spawn(b.handle("windowd").unwrap(), "windowcheck", boot, u64::MAX)
            .unwrap()
    };
    let (check, observe) = sys::channel_create().unwrap();
    let game_boot = Builder::default()
        .handle("window", client)
        .handle("check", observe)
        .send()
        .unwrap();
    let doom = sys::spawn_linux(
        b.handle("doom").unwrap(),
        &[
            "oscar-doom",
            "-iwad",
            b.value("iwad").unwrap_or("/freedoom1.wad"),
            "-warp",
            "1",
            "1",
            "-nomonsters",
        ],
        &[],
        Some(game_boot),
    )
    .unwrap();
    let hash = || {
        let mut h = 0u64;
        for y in 80..450 {
            for x in 60..650 {
                let pixel = unsafe {
                    (reader as *const u32)
                        .add(y * stride as usize + x)
                        .read_volatile()
                } & 0xffffff;
                h = h.wrapping_mul(0x100000001b3) ^ pixel as u64;
            }
        }
        h
    };
    let state = || {
        let mut bytes = [0u8; 20];
        let mut handles = [Handle(0); 1];
        let (n, nh) = sys::channel_read(check, &mut bytes, &mut handles, sys::after_ms(10000))
            .expect("Doom did not enter a level");
        assert_eq!((n, nh), (20, 0));
        core::array::from_fn::<i32, 5, _>(|i| {
            i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        })
    };
    let start = state();
    let mut hashes = std::collections::BTreeSet::new();
    for _ in 0..120 {
        pause(100);
        hashes.insert(hash());
        if hashes.len() >= 2 {
            break;
        }
    }
    assert!(hashes.len() >= 2, "Doom framebuffer did not advance");
    println!("windowcheck: Doom renders changing frames in an Oscar window");
    packet(
        input,
        Input::KeyboardState {
            modifiers: 1,
            keys: [82, 0, 0, 0, 0, 0],
        },
    );
    let mut moved = false;
    for _ in 0..100 {
        let latest = state();
        if latest[1] > start[1] + 10 && (latest[2], latest[3]) != (start[2], start[3]) {
            moved = true;
            break;
        }
    }
    assert!(moved, "Doom did not respond to held forward input");
    println!("windowcheck: playable Doom level advances and held forward input moves the player");
    packet(
        input,
        Input::KeyboardState {
            modifiers: 0,
            keys: [0; 6],
        },
    );
    // A second native shared window occludes Doom and receives focused raw input.
    let mut probe = Window::open(testclient, 120, 80, "Check").unwrap();
    let mut pixels = Canvas::new(120, 80).unwrap();
    pixels.pixels_mut().fill(0x12ab34);
    probe.present(pixels.pixels()).unwrap();
    let mut ready = false;
    let mut input_seen = false;
    packet(
        input,
        Input::KeyboardState {
            modifiers: 0,
            keys: [4, 0, 0, 0, 0, 0],
        },
    );
    let until = sys::after_ms(2000);
    while sys::clock() < until {
        match probe.poll(sys::after_ms(100)) {
            Ok(Some(Event::Input(Input::KeyboardState { keys, .. }))) => {
                input_seen |= keys[0] == 4;
            }
            Ok(_) => {}
            Err(Error::TimedOut | Error::ShouldWait) => {}
            Err(e) => panic!("probe: {e:?}"),
        }
        ready |= probe.can_present();
        if ready && input_seen {
            break;
        }
    }
    assert!(
        ready && input_seen,
        "shared frame/input acknowledgements missing"
    );
    if width >= 866 {
        packet(
            input,
            Input::Pointer {
                x: 760,
                y: 50,
                down: true,
            },
        );
        packet(
            input,
            Input::Pointer {
                x: 124,
                y: 90,
                down: true,
            },
        );
        packet(
            input,
            Input::Pointer {
                x: 124,
                y: 90,
                down: false,
            },
        );
    }
    pause(100);
    let mut found = false;
    for i in 0..(width * height) as usize {
        if unsafe {
            (reader as *const u32)
                .add(i / width as usize * stride as usize + i % width as usize)
                .read_volatile()
        } & 0xffffff
            == 0x12ab34
        {
            found = true;
            break;
        }
    }
    assert!(found, "occluding shared surface missing");
    println!("windowcheck: capture native Doom and occluding window");
    pause(500);
    drop(probe);
    pause(100);
    println!("windowcheck: shared surface, focus/input and occlusion passed");
    // Drag Doom using the trusted seat, then request close through its title box.
    packet(
        input,
        Input::Pointer {
            x: 110,
            y: 50,
            down: true,
        },
    );
    packet(
        input,
        Input::Pointer {
            x: 150,
            y: 90,
            down: true,
        },
    );
    packet(
        input,
        Input::Pointer {
            x: 150,
            y: 90,
            down: false,
        },
    );
    pause(100);
    packet(
        input,
        Input::Pointer {
            x: 78 + 40 + 640 + 6 - 13,
            y: 42 + 40 + 10,
            down: true,
        },
    );
    packet(
        input,
        Input::Pointer {
            x: 78 + 40 + 640 + 6 - 13,
            y: 42 + 40 + 10,
            down: false,
        },
    );
    let code = sys::process_wait(doom, sys::after_ms(3000)).expect("Doom did not close");
    assert_eq!(code, 0);
    let _ = sys::close(control);
    assert_eq!(sys::process_wait(windowd, sys::after_ms(2000)).unwrap(), 0);
    let _ = sys::close(doom);
    let _ = sys::close(windowd);
    let _ = sys::close(input);
    let _ = sys::close(check);
    let _ = sys::vm_unmap(reader, size);
    let _ = sys::close(framebuffer);
    println!("windowcheck: Doom closes and windowd releases the session");
}
