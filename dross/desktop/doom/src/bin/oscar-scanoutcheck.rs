//! Physical scanout acceptance: no assertions about the old framebuffer address.
//! Requires external camera evidence plus windowd's observed flip/restore logs.
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
fn send(ch: Handle, input: Input) {
    sys::channel_write(
        ch,
        &Packet::new(0, 0, Message::Input(input)).encode().unwrap(),
        &[],
    )
    .unwrap();
}
fn pointer(ch: Handle, x: i32, y: i32, down: bool) {
    send(ch, Input::Pointer { x, y, down });
}
fn keys(ch: Handle, modifiers: u8, keys: [u8; 6]) {
    send(ch, Input::KeyboardState { modifiers, keys });
}
fn pause(ms: u64) {
    std::thread::sleep(std::time::Duration::from_millis(ms));
}
fn state(channel: Handle, deadline: u64) -> [i32; 5] {
    let mut latest = None;
    for _ in 0..256 {
        let mut bytes = [0; 20];
        let mut handles = [Handle(0); 64];
        match sys::channel_read(
            channel,
            &mut bytes,
            &mut handles,
            if latest.is_none() { deadline } else { 0 },
        ) {
            Ok((n, nh)) => {
                for &h in &handles[..nh] {
                    let _ = sys::close(h);
                }
                assert_eq!((n, nh), (20, 0), "invalid Doom state");
                latest = Some(core::array::from_fn(|i| {
                    i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
                }));
            }
            Err(Error::ShouldWait | Error::TimedOut) if latest.is_some() => break,
            Err(error) => panic!("Doom state: {error:?}"),
        }
    }
    latest.unwrap()
}
fn main() {
    let b = Bootstrap::read(Handle(1)).expect("scanout acceptance bootstrap");
    let _ = sys::close(Handle(1));
    let (input, seat) = sys::channel_create().unwrap();
    let (control, lifetime) = sys::channel_create().unwrap();
    let (server, client) = sys::channel_create().unwrap();
    let (probe_server, probe_client) = sys::channel_create().unwrap();
    let vulkan = b
        .handle("vkdraw")
        .map(|image| (image, sys::channel_create().unwrap()));
    let mut boot = Builder::default()
        .handle("display", b.handle("display").unwrap())
        .handle("scanout", b.handle("scanout").unwrap())
        .handle("input", seat)
        .handle("control", lifetime)
        .handle("client/doom", server)
        .handle("client/check", probe_server);
    if let Some((_, (server, _))) = vulkan {
        boot = boot.handle("client/vulkan", server);
    }
    for name in ["width", "height", "stride", "format", "offset", "gpu-bus"] {
        if let Some(value) = b.value(name) {
            boot = boot.value(name, value);
        }
    }
    let mut environment: Vec<String> = [
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "VK_DRIVER_FILES",
        "GPUD_SOCKET",
    ]
    .iter()
    .filter_map(|name| b.value(name).map(|value| format!("{name}={value}")))
    .collect();
    environment.push("DROSS_BOOTSTRAP=1".into());
    let env: Vec<_> = environment.iter().map(String::as_str).collect();
    let boot = boot.send().unwrap();
    let windowd = sys::spawn_linux(
        b.handle("windowd-gpu").unwrap(),
        &["oscar-windowd-gpu"],
        &env,
        Some(boot),
    )
    .unwrap();
    let (check, observer) = sys::channel_create().unwrap();
    let boot = Builder::default()
        .handle("window", client)
        .handle("check", observer)
        .send()
        .unwrap();
    let doom = sys::spawn_linux(
        b.handle("doom").unwrap(),
        &[
            "oscar-doom",
            "-iwad",
            b.value("iwad").unwrap(),
            "-warp",
            "1",
            "1",
            "-nomonsters",
            "-nosound",
        ],
        &[],
        Some(boot),
    )
    .unwrap();
    let first = state(check, sys::after_ms(15000));
    keys(input, 1, [82, 0, 0, 0, 0, 0]);
    let deadline = sys::after_ms(5000);
    loop {
        let current = state(check, deadline);
        if current[1] > first[1] + 10 && (current[2], current[3]) != (first[2], first[3]) {
            break;
        }
        assert!(sys::clock() < deadline, "held forward did not move player");
    }
    keys(input, 0, [0; 6]);
    println!("scanoutcheck: real Doom level advances and held keyboard input moves player");
    let mut probe = Window::open(probe_client, 120, 80, "Check").unwrap();
    let mut canvas = Canvas::new(120, 80).unwrap();
    canvas.pixels_mut().fill(0x12ab34);
    probe.present(canvas.pixels()).unwrap();
    keys(input, 0, [4, 0, 0, 0, 0, 0]);
    let deadline = sys::after_ms(3000);
    let mut focused_input = false;
    while !probe.can_present() || !focused_input {
        match probe.poll(deadline) {
            Ok(Some(Event::Input(Input::KeyboardState { keys, .. }))) => {
                focused_input |= keys[0] == 4
            }
            Ok(_) => {}
            Err(error) => panic!("native probe frame/focus: {error:?}"),
        }
        assert!(sys::clock() < deadline, "probe not acknowledged");
    }
    keys(input, 0, [0; 6]);
    println!(
        "scanoutcheck: second native surface acknowledged and receives focused keyboard state"
    );
    // Put the probe over Doom, then expose the game for the camera capture.
    if b.value_u64("width").unwrap_or(0) >= 866 {
        pointer(input, 760, 50, true);
        pointer(input, 124, 90, true);
        pointer(input, 124, 90, false);
    }
    pause(1000);
    drop(probe);
    // Focusing Doom restores actual held/released state. Turn, then release;
    // unlike forward motion, angle has no friction/inertia after key release.
    pointer(input, 120, 110, true);
    pointer(input, 120, 110, false);
    let before_turn = state(check, sys::after_ms(2000));
    keys(input, 0, [79, 0, 0, 0, 0, 0]);
    let deadline = sys::after_ms(3000);
    loop {
        let current = state(check, deadline);
        if current[4] != before_turn[4] {
            break;
        }
        assert!(sys::clock() < deadline, "held turn did not change angle");
    }
    keys(input, 0, [0; 6]);
    pause(350);
    let released = state(check, sys::after_ms(2000));
    pause(350);
    let stopped = state(check, sys::after_ms(2000));
    assert_eq!(stopped[4], released[4], "turn remained held after release");
    println!("scanoutcheck: actual held/released keyboard state changes and stops turn");
    let vulkan = vulkan.map(|(image, (_, client))| {
        let (check, observer) = sys::channel_create().unwrap();
        let boot = Builder::default().handle("window", client).handle("check", observer)
            .value("gpu-bus", b.value("gpu-bus").unwrap_or("6")).send().unwrap();
        let process = sys::spawn_linux(image, &["oscar-vkdraw"], &env, Some(boot)).unwrap();
        let read = |deadline| {
            let mut bytes = [0u8; 20]; let mut handles = [Handle(0); 64];
            let (n, nh) = sys::channel_read(check, &mut bytes, &mut handles, deadline).expect("native Vulkan app did not acknowledge frame");
            for &h in &handles[..nh] { let _ = sys::close(h); }
            assert_eq!((n, nh), (20, 0));
            assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 0xf3c54d);
            (u64::from_le_bytes(bytes[..8].try_into().unwrap()), i32::from_le_bytes(bytes[8..12].try_into().unwrap()), i32::from_le_bytes(bytes[12..16].try_into().unwrap()))
        };
        assert!(b.value_u64("width").unwrap_or(0) >= 1230, "Vulkan/Doom acceptance needs room for both windows");
        let first = read(sys::after_ms(15000));
        pointer(input, 900, 170, true); pointer(input, 910, 180, false);
        let deadline = sys::after_ms(5000);
        loop {
            let current = read(deadline);
            if current.0 > first.0 && (current.1, current.2) != (first.1, first.2) {
                assert!((0..480).contains(&current.1) && (0..320).contains(&current.2)); break; }
            assert!(sys::clock() < deadline, "native Vulkan app did not respond to pointer input");
        }
        println!("scanoutcheck: Vulkan app draws exact GPU pixels into native Window Memory, receives pointer input and frame acknowledgements");
        // Return focus to Doom while keeping the GPU app visible beside it.
        pointer(input, 120, 110, true); pointer(input, 120, 110, false);
        (process, check)
    });
    keys(input, 1, [0; 6]);
    println!("scanoutcheck: capture visible Doom firing for 10 seconds");
    let mut previous = stopped;
    for second in 1..=10 {
        pause(1000);
        let current = state(check, sys::after_ms(2000));
        assert!(current[1] > previous[1], "Doom stopped while presenting");
        previous = current;
        println!(
            "scanoutcheck: visible {second}/10, level tic {}",
            current[1]
        );
    }
    if let Some((process, check)) = vulkan {
        pointer(input, 1213, 52, true);
        pointer(input, 1213, 52, false);
        assert_eq!(
            sys::process_wait(process, sys::after_ms(5000))
                .expect("native Vulkan app did not close"),
            0
        );
        let _ = sys::close(process);
        let _ = sys::close(check);
        println!("scanoutcheck: native Vulkan app closed and released its shared surface");
    }
    keys(input, 0, [0; 6]);
    // Known initial placement plus an explicit 40-pixel drag. Existing scene
    // tests check the exact policy/pixels; here observe the real display by camera.
    pointer(input, 110, 50, true);
    pointer(input, 150, 90, true);
    pointer(input, 150, 90, false);
    pause(700);
    pointer(input, 751, 92, true);
    pointer(input, 751, 92, false);
    assert_eq!(
        sys::process_wait(doom, sys::after_ms(5000)).expect("Doom did not close"),
        0
    );
    let _ = sys::close(control);
    assert_eq!(
        sys::process_wait(windowd, sys::after_ms(5000)).expect("windowd did not restore firmware"),
        0
    );
    for handle in [doom, windowd, input, check] {
        let _ = sys::close(handle);
    }
    println!(
        "scanoutcheck: Doom closed, windowd exited successfully after firmware restore acknowledgement"
    );
}
