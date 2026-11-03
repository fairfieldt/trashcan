use oscar_desktop::{
    app::Trash,
    canvas::{Canvas, Rect},
    protocol::*,
    server::Server,
    shell::{self, Environment, Shell},
    terminal::{self, Terminal},
    ui::Ui,
};

#[derive(Default)]
struct Files;
impl Environment for Files {
    fn pwd(&self) -> String {
        "/".into()
    }
    fn cd(&mut self, _: &str) -> Result<(), String> {
        Ok(())
    }
    fn list(&mut self, _: &str) -> Result<String, String> {
        Ok("notes".into())
    }
    fn read(&mut self, path: &str) -> Result<String, String> {
        if path == "notes" {
            Ok("hello from a file".into())
        } else {
            Err("file not found".into())
        }
    }
}
fn create(s: &mut Server, client: u32, width: u32, height: u32) -> u32 {
    let out = s.request(
        client,
        Packet::new(
            1,
            0,
            Message::Create {
                width,
                height,
                title: "Test".into(),
            },
        ),
    );
    assert!(matches!(out[0].packet.message, Message::Created { .. }));
    out[0].packet.window
}
fn reply(
    s: &mut Server,
    client: u32,
    window: u32,
    message: Message,
) -> Vec<oscar_desktop::server::Delivery> {
    s.request(client, Packet::new(2, window, message))
}

#[test]
fn wire_round_trips_and_rejects_all_truncations() {
    let messages = [
        Message::Create {
            width: 120,
            height: 80,
            title: "Trash".into(),
        },
        Message::Created {
            width: 120,
            height: 80,
        },
        Message::Upload {
            offset: 2,
            pixels: vec![0x112233, 0xffffff],
        },
        Message::Present,
        Message::FrameDone,
        Message::Close,
        Message::Closed,
        Message::CloseRequested,
        Message::Focus(true),
        Message::Error(Fault::Ownership),
        Message::Input(Input::Pointer {
            x: -5,
            y: 100,
            down: true,
        }),
        Message::Input(Input::Key {
            key: Key::Character('é'),
            mods: Mods(3),
        }),
    ];
    for message in messages {
        let packet = Packet::new(7, 13, message);
        let bytes = packet.encode().unwrap();
        assert_eq!(Packet::decode(&bytes), Ok(packet));
        for len in 0..bytes.len() {
            assert!(Packet::decode(&bytes[..len]).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Packet::decode(&trailing).is_err());
    }
}
#[test]
fn wire_rejects_versions_scalars_flags_and_oversize() {
    let mut bytes = Packet::new(1, 1, Message::Present).encode().unwrap();
    bytes[4] = 2;
    assert_eq!(Packet::decode(&bytes), Err(Fault::Version));
    let mut bytes = Packet::new(
        1,
        1,
        Message::Input(Input::Key {
            key: Key::Character('a'),
            mods: Mods(0),
        }),
    )
    .encode()
    .unwrap();
    bytes[HEADER..HEADER + 4].copy_from_slice(&0xd800u32.to_le_bytes());
    assert!(Packet::decode(&bytes).is_err());
    bytes[HEADER..HEADER + 4].copy_from_slice(&65u32.to_le_bytes());
    bytes[HEADER + 4] = 128;
    assert!(Packet::decode(&bytes).is_err());
    assert!(
        Packet::new(
            1,
            1,
            Message::Upload {
                offset: 0,
                pixels: vec![0; MAX_PACKET]
            }
        )
        .encode()
        .is_err()
    );
}
#[test]
fn wire_random_inputs_never_panic() {
    let mut n = 7u64;
    for size in 0..1024 {
        let mut bytes = vec![0; size];
        for b in &mut bytes {
            n ^= n << 13;
            n ^= n >> 7;
            n ^= n << 17;
            *b = n as u8;
        }
        if size >= HEADER {
            bytes[..4].copy_from_slice(b"OSCR");
            bytes[4] = 1;
            bytes[5] = 0;
            bytes[16..20].copy_from_slice(&((size - HEADER) as u32).to_le_bytes());
        }
        let _ = Packet::decode(&bytes);
    }
}
#[test]
fn clients_cannot_mutate_another_window_or_inject_input() {
    let mut s = Server::new(640, 480).unwrap();
    let id = create(&mut s, 1, 200, 100);
    for message in [
        Message::Close,
        Message::Present,
        Message::Upload {
            offset: 0,
            pixels: vec![0xff0000],
        },
    ] {
        assert_eq!(
            reply(&mut s, 2, id, message)[0].packet.message,
            Message::Error(Fault::Ownership)
        );
    }
    assert_eq!(
        reply(
            &mut s,
            1,
            id,
            Message::Input(Input::Key {
                key: Key::Enter,
                mods: Mods(0)
            })
        )[0]
        .packet
        .message,
        Message::Error(Fault::Unknown)
    );
    assert_eq!(s.window_count(), 1);
}
#[test]
fn upload_is_atomic_at_present_and_bounds_are_checked() {
    let mut s = Server::new(640, 480).unwrap();
    let id = create(&mut s, 1, 200, 100);
    let before = s.render().pixels().to_vec();
    assert!(
        reply(
            &mut s,
            1,
            id,
            Message::Upload {
                offset: 0,
                pixels: vec![0xff0000; 20000]
            }
        )
        .is_empty()
    );
    assert_eq!(s.render().pixels(), before);
    assert_eq!(
        reply(
            &mut s,
            1,
            id,
            Message::Upload {
                offset: u32::MAX,
                pixels: vec![1]
            }
        )[0]
        .packet
        .message,
        Message::Error(Fault::Geometry)
    );
    let out = reply(&mut s, 1, id, Message::Present);
    assert_eq!(out[0].packet.serial, 2);
    assert_eq!(out[0].packet.message, Message::FrameDone);
    assert_ne!(s.render().pixels(), before);
}
#[test]
fn quota_disconnect_and_stale_ids() {
    let mut s = Server::new(1024, 768).unwrap();
    let id = create(&mut s, 1, 760, 500);
    let out = s.request(
        2,
        Packet::new(
            1,
            0,
            Message::Create {
                width: 760,
                height: 500,
                title: "Large".into(),
            },
        ),
    );
    assert_eq!(out[0].packet.message, Message::Error(Fault::Limit));
    s.disconnect(1);
    assert_eq!(s.window_count(), 0);
    assert_eq!(s.focused(), None);
    let next = create(&mut s, 2, 760, 500);
    assert_ne!(next, id);
    assert_eq!(
        reply(&mut s, 1, id, Message::Present)[0].packet.message,
        Message::Error(Fault::Ownership)
    );
}
#[test]
fn focus_cycle_drag_close_and_pointer_capture() {
    let mut s = Server::new(640, 480).unwrap();
    let a = create(&mut s, 1, 200, 100);
    let b = create(&mut s, 2, 200, 100);
    assert_eq!(s.focused(), Some(b));
    let out = s.input(Input::Key {
        key: Key::Tab,
        mods: Mods(Mods::ALT),
    });
    assert_eq!(s.focused(), Some(a));
    assert!(
        out.iter()
            .any(|d| d.client == 1 && d.packet.message == Message::Focus(true))
    );
    s.input(Input::Pointer {
        x: 90,
        y: 50,
        down: true,
    });
    s.input(Input::Pointer {
        x: 400,
        y: 250,
        down: true,
    });
    s.input(Input::Pointer {
        x: 400,
        y: 250,
        down: false,
    });
    // First window was dragged by its title to (388,242).
    let out = s.input(Input::Pointer {
        x: 580,
        y: 250,
        down: true,
    });
    assert!(
        out.iter()
            .any(|d| d.packet.window == a && d.packet.message == Message::CloseRequested)
    );
    s.input(Input::Pointer {
        x: 580,
        y: 250,
        down: false,
    });
    let out = s.input(Input::Pointer {
        x: 400,
        y: 280,
        down: true,
    });
    assert!(out.iter().any(|d| matches!(
        d.packet.message,
        Message::Input(Input::Pointer { down: true, .. })
    )));
    let out = s.input(Input::Pointer {
        x: 10,
        y: 10,
        down: false,
    });
    assert!(out.iter().any(|d| d.packet.window == a && matches!(d.packet.message, Message::Input(Input::Pointer { x, down: false, .. }) if x < 0)));
}
#[test]
fn canvas_clips_completely_disjoint_and_extreme_rectangles() {
    let mut c = Canvas::new(10, 10).unwrap();
    for r in [
        Rect::new(100, 0, 4, 4),
        Rect::new(-100, 0, 4, 4),
        Rect::new(i32::MAX, i32::MAX, u32::MAX, u32::MAX),
    ] {
        c.fill(r, 1);
    }
    c.blit(&Canvas::new(5, 5).unwrap(), 100, 0);
    assert!(c.pixels().iter().all(|&p| p == 0));
    c.fill(Rect::new(-2, -2, 4, 4), 0xff00);
    assert_eq!(c.pixels().iter().filter(|&&p| p == 0xff00).count(), 4);
}
#[test]
fn widget_requires_press_and_release_on_same_button() {
    let mut c = Canvas::new(200, 100).unwrap();
    let mut ui = Ui::default();
    let r = Rect::new(10, 10, 80, 24);
    ui.begin(Some(Input::Pointer {
        x: 20,
        y: 20,
        down: true,
    }));
    assert!(!ui.button(&mut c, 1, r, "Clear"));
    ui.end();
    ui.begin(Some(Input::Pointer {
        x: 190,
        y: 90,
        down: false,
    }));
    assert!(!ui.button(&mut c, 1, r, "Clear"));
    ui.end();
    ui.begin(Some(Input::Pointer {
        x: 20,
        y: 20,
        down: true,
    }));
    ui.button(&mut c, 1, r, "Clear");
    ui.end();
    ui.begin(Some(Input::Pointer {
        x: 20,
        y: 20,
        down: false,
    }));
    assert!(ui.button(&mut c, 1, r, "Clear"));
    ui.end();
    ui.begin(Some(Input::Key {
        key: Key::Enter,
        mods: Mods(0),
    }));
    assert!(ui.button(&mut c, 1, r, "Clear"));
    ui.end();
}
#[test]
fn terminal_chunk_boundaries_do_not_change_output() {
    let bytes = b"hello\r\n\x1b[31;1mred\x1b[0m \xc3\xa9\x1b[2;3H!\x1b]0;ignored\x07ok";
    let mut full = Terminal::new(20, 4).unwrap();
    full.feed(bytes);
    for chunk in 1..=bytes.len() {
        let mut t = Terminal::new(20, 4).unwrap();
        for bytes in bytes.chunks(chunk) {
            t.feed(bytes);
        }
        assert_eq!(t.cells(), full.cells());
        assert_eq!(t.cursor(), full.cursor());
    }
}
#[test]
fn terminal_wrap_erase_color_and_alternate_screen() {
    let mut t = Terminal::new(5, 3).unwrap();
    t.feed(b"12345");
    assert_eq!(t.cursor(), (4, 0));
    t.feed(b"X");
    assert_eq!(t.line(1), "X    ");
    t.feed(b"\x1b[2;2H\x1b[38;2;1;2;3mY\x1b[K");
    assert_eq!(t.cells()[6].fg, 0x010203);
    let saved = t.cells().to_vec();
    let cursor = t.cursor();
    t.feed(b"\x1b[?1049hother\x1b[?1049l");
    assert_eq!(t.cells(), saved);
    assert_eq!(t.cursor(), cursor);
    t.feed(b"\x1b[2J");
    assert!(t.cells().iter().all(|c| c.ch == ' '));
}
#[test]
fn terminal_scroll_regions_and_bounded_history() {
    let mut t = Terminal::new(8, 4).unwrap();
    t.feed(b"top\x1b[4;1Hbottom\x1b[2;3r\x1b[3;1Hx\r\ny");
    assert!(t.line(0).starts_with("top"));
    assert!(t.line(3).starts_with("bottom"));
    assert_eq!(t.history_len(), 0);
    t.feed(b"\x1b[r");
    for _ in 0..1000 {
        t.feed(b"line\r\n");
    }
    assert_eq!(t.history_len(), 256);
    t.scroll_view(10000);
    t.draw(
        &mut Canvas::new(100, 100).unwrap(),
        Rect::new(0, 0, 100, 100),
        true,
    );
}
#[test]
fn terminal_escape_flood_and_invalid_utf8_recover() {
    let mut t = Terminal::new(20, 4).unwrap();
    t.feed(b"\x1b]");
    t.feed(&vec![b'x'; 100000]);
    t.feed(b"\x1b\\OK\xff\xc3!");
    assert!(t.line(0).starts_with("OK\u{fffd}\u{fffd}!"));
    t.feed(b"\x1b[");
    t.feed(&vec![b'9'; 10000]);
    t.feed(b"mZ");
    assert!(t.line(0).contains('Z'));
}
#[test]
fn terminal_key_encoding() {
    assert_eq!(
        terminal::encode_key(Key::Character('c'), Mods(Mods::CTRL)),
        b"\x03"
    );
    assert_eq!(terminal::encode_key(Key::Up, Mods(0)), b"\x1b[A");
    assert_eq!(
        terminal::encode_key(Key::Character('é'), Mods(Mods::ALT)),
        b"\x1b\xc3\xa9"
    );
}
#[test]
fn shell_quotes_errors_and_empty_arguments() {
    assert_eq!(
        shell::parse("echo 'a b' \"\" c\\ d").unwrap(),
        ["echo", "a b", "", "c d"]
    );
    for line in [
        "echo 'unfinished",
        "echo \\",
        "echo hi | cat",
        "echo hi; exit",
    ] {
        assert!(shell::parse(line).is_err());
    }
}
#[test]
fn shell_editing_history_and_files() {
    let mut t = Terminal::new(80, 20).unwrap();
    let mut shell = Shell::default();
    let mut env = Files;
    shell.start(&mut t);
    for c in "echo aé".chars() {
        shell.key(Key::Character(c), Mods(0), &mut t, &mut env);
    }
    shell.key(Key::Left, Mods(0), &mut t, &mut env);
    shell.key(Key::Backspace, Mods(0), &mut t, &mut env);
    assert_eq!(shell.line(), "echo é");
    shell.key(Key::Enter, Mods(0), &mut t, &mut env);
    shell.key(Key::Up, Mods(0), &mut t, &mut env);
    assert_eq!(shell.line(), "echo é");
    shell.key(Key::Down, Mods(0), &mut t, &mut env);
    assert_eq!(shell.line(), "");
    shell.command("cat notes", &mut t, &mut env);
    assert!((0..20).any(|row| t.line(row).contains("hello from a file")));
}
#[test]
fn complete_session_uses_wire_upload_ack_and_disconnect() {
    let mut server = Server::new(1024, 768).unwrap();
    let mut app = Trash::new(760, 500).unwrap();
    let mut env = Files;
    let wire = app.create().encode().unwrap();
    for d in server.request(1, Packet::decode(&wire).unwrap()) {
        app.receive(
            Packet::decode(&d.packet.encode().unwrap()).unwrap(),
            &mut env,
        )
        .unwrap();
    }
    for c in "echo works".chars() {
        for d in server.input(Input::Key {
            key: Key::Character(c),
            mods: Mods(0),
        }) {
            app.receive(d.packet, &mut env).unwrap();
        }
    }
    for d in server.input(Input::Key {
        key: Key::Enter,
        mods: Mods(0),
    }) {
        app.receive(d.packet, &mut env).unwrap();
    }
    assert!(app.terminal.line(1).starts_with("works"));
    let mut out = Vec::new();
    app.frame(&mut env, |packet| {
        out.extend(server.request(1, Packet::decode(&packet.encode()?)?));
        Ok(())
    })
    .unwrap();
    assert!(out.iter().any(|d| d.packet.message == Message::FrameDone));
    // No second frame before the acknowledgement.
    app.dirty = true;
    app.frame(&mut env, |_| panic!("must apply backpressure"))
        .unwrap();
    for d in out {
        app.receive(d.packet, &mut env).unwrap();
    }
    assert!(server.render().pixels().contains(&0xf3c54d));
    server.request(1, app.close_packet());
    assert_eq!(server.window_count(), 0);
}

#[test]
fn widget_tab_order_wraps_in_both_directions() {
    let mut c = Canvas::new(200, 100).unwrap();
    let mut ui = Ui::default();
    let mut paint = |ui: &mut Ui, input| {
        ui.begin(input);
        for id in 1..=3 {
            ui.button(&mut c, id, Rect::new(id as i32 * 50, 10, 40, 24), "Go");
        }
        ui.end();
    };
    let tab = |shift| {
        Some(Input::Key {
            key: Key::Tab,
            mods: Mods(if shift { Mods::SHIFT } else { 0 }),
        })
    };
    paint(&mut ui, tab(false));
    assert_eq!(ui.focused(), Some(1));
    paint(&mut ui, tab(false));
    assert_eq!(ui.focused(), Some(2));
    paint(&mut ui, tab(false));
    assert_eq!(ui.focused(), Some(3));
    paint(&mut ui, tab(false));
    assert_eq!(ui.focused(), Some(1));
    paint(&mut ui, tab(true));
    assert_eq!(ui.focused(), Some(3));
    paint(&mut ui, tab(true));
    assert_eq!(ui.focused(), Some(2));
}

#[test]
fn scrollbar_pointer_capture_and_keyboard() {
    let mut c = Canvas::new(100, 200).unwrap();
    let mut ui = Ui::default();
    let r = Rect::new(10, 10, 18, 100);
    ui.begin(Some(Input::Pointer {
        x: 20,
        y: 10,
        down: true,
    }));
    let value = ui.scrollbar(&mut c, 1, r, 100, 50, 20);
    ui.end();
    assert_eq!(value, 0);
    ui.begin(Some(Input::Pointer {
        x: 90,
        y: 190,
        down: true,
    }));
    let value = ui.scrollbar(&mut c, 1, r, 100, value, 20);
    ui.end();
    assert_eq!(value, 100);
    ui.begin(Some(Input::Pointer {
        x: 90,
        y: 190,
        down: false,
    }));
    ui.scrollbar(&mut c, 1, r, 100, value, 20);
    ui.end();
    ui.begin(Some(Input::Key {
        key: Key::PageUp,
        mods: Mods(0),
    }));
    assert_eq!(ui.scrollbar(&mut c, 1, r, 100, value, 20), 80);
    ui.end();
}

#[test]
fn byte_stream_client_does_not_run_trash_commands() {
    let mut app = Trash::new(760, 500).unwrap();
    let mut env = Files;
    app.use_byte_stream();
    app.input(
        Input::Key {
            key: Key::Character('c'),
            mods: Mods(Mods::CTRL),
        },
        &mut env,
    );
    assert_eq!(app.take_input(), b"\x03");
    assert!(app.take_input().is_empty());
    app.feed(b"\x1b[32mexternal peer\x1b[0m\r\n");
    assert!(app.terminal.line(0).starts_with("external peer"));
    assert_eq!(app.shell.line(), "");
}
