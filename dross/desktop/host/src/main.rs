//! Host seat/display adapter. Protocol serialization is used even in this process.
use minifb::{
    InputCallback, Key as HostKey, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions,
};
use oscar_desktop::{
    app::{HEIGHT, Trash, WIDTH},
    canvas::Canvas,
    protocol::{Input, Key, Mods, Packet},
    server::{Delivery, Server},
    shell::Environment,
};
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender},
};

struct Files {
    cwd: PathBuf,
}
impl Files {
    fn path(&self, path: &str) -> PathBuf {
        self.cwd.join(path)
    }
}
impl Environment for Files {
    fn pwd(&self) -> String {
        self.cwd.display().to_string()
    }
    fn cd(&mut self, path: &str) -> Result<(), String> {
        let path = self.path(path).canonicalize().map_err(|e| e.to_string())?;
        if !path.is_dir() {
            return Err("not a directory".into());
        }
        self.cwd = path;
        Ok(())
    }
    fn list(&mut self, path: &str) -> Result<String, String> {
        let mut names = Vec::new();
        for entry in fs::read_dir(self.path(path))
            .map_err(|e| e.to_string())?
            .take(512)
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry
                .file_name()
                .to_string_lossy()
                .replace(|c: char| c.is_control(), "?");
            names.push(format!(
                "{name}{}",
                if entry.path().is_dir() { "/" } else { "" }
            ));
        }
        names.sort();
        Ok(names.join("\n"))
    }
    fn read(&mut self, path: &str) -> Result<String, String> {
        if !fs::metadata(self.path(path))
            .map_err(|e| e.to_string())?
            .is_file()
        {
            return Err("not a regular file".into());
        }
        let file = fs::File::open(self.path(path)).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take(32 * 1024)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        String::from_utf8(bytes).map_err(|_| "file is not UTF-8".into())
    }
}

fn dispatch(server: &mut Server, app: &mut Trash, env: &mut Files, packet: Packet) {
    let bytes = packet.encode().expect("encode client packet");
    let out = server.request(1, Packet::decode(&bytes).expect("decode client packet"));
    deliver(app, env, out);
}
fn deliver(app: &mut Trash, env: &mut Files, out: Vec<Delivery>) {
    for delivery in out {
        assert_eq!(delivery.client, 1);
        let bytes = delivery.packet.encode().expect("encode server packet");
        app.receive(Packet::decode(&bytes).expect("decode server packet"), env)
            .expect("server response");
    }
}
fn frame(server: &mut Server, app: &mut Trash, env: &mut Files) {
    let mut replies = Vec::new();
    app.frame(env, |packet| {
        let wire = packet.encode()?;
        replies.extend(server.request(1, Packet::decode(&wire)?));
        Ok(())
    })
    .expect("upload frame");
    deliver(app, env, replies);
}
fn ppm(canvas: &Canvas, path: &str) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(fs::File::create(path)?);
    write!(file, "P6\n{} {}\n255\n", canvas.width(), canvas.height())?;
    for &pixel in canvas.pixels() {
        file.write_all(&[(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8])?;
    }
    file.flush()
}
struct TextInput(SyncSender<char>);
impl InputCallback for TextInput {
    fn add_char(&mut self, c: u32) {
        if let Some(c) = char::from_u32(c).filter(|c| !c.is_control()) {
            let _ = self.0.try_send(c);
        }
    }
}

struct Pty {
    input: Box<dyn Write + Send>,
    output: Receiver<Vec<u8>>,
    child: Box<dyn portable_pty::Child + Send + Sync>,
    _master: Box<dyn portable_pty::MasterPty + Send>,
}
impl Pty {
    fn spawn(argv: &[String], cols: usize, rows: usize) -> Self {
        let pair = native_pty_system()
            .openpty(PtySize {
                rows: rows as u16,
                cols: cols as u16,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("open PTY");
        let mut command = CommandBuilder::new(&argv[0]);
        command.args(&argv[1..]);
        command.env("TERM", "xterm-256color");
        let child = pair
            .slave
            .spawn_command(command)
            .expect("spawn terminal command");
        drop(pair.slave);
        let mut reader = pair.master.try_clone_reader().expect("PTY reader");
        let input = pair.master.take_writer().expect("PTY writer");
        let (tx, output) = mpsc::sync_channel(16);
        std::thread::spawn(move || {
            let mut bytes = [0; 4096];
            loop {
                match reader.read(&mut bytes) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send(bytes[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Self {
            input,
            output,
            child,
            _master: pair.master,
        }
    }
}
impl Drop for Pty {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--help") {
        println!(
            "oscar-desktop-host [--headless FILE.ppm [COMMAND...]]\noscar-desktop-host --exec PROGRAM [ARG...]\nCtrl-Tab: switch between terminal and controls. Alt-Tab: cycle windows."
        );
        return;
    }
    let headless = args.first().is_some_and(|s| s == "--headless");
    let external = args.first().is_some_and(|s| s == "--exec");
    if (headless || external) && args.len() < 2 {
        eprintln!("missing output path or program; use --help");
        std::process::exit(2);
    }
    let mut server = Server::new(1024, 768).unwrap();
    let mut app = Trash::new(WIDTH, HEIGHT).unwrap();
    let mut env = Files {
        cwd: std::env::current_dir().unwrap(),
    };
    let create = app.create();
    dispatch(&mut server, &mut app, &mut env, create);
    if headless {
        let defaults = ["help".to_string(), "echo 'Oscar'".to_string()];
        let commands = if args.len() > 2 {
            &args[2..]
        } else {
            &defaults
        };
        for command in commands {
            app.command(command, &mut env);
        }
        frame(&mut server, &mut app, &mut env);
        ppm(server.render(), &args[1]).expect("write snapshot");
        for row in 0..app.terminal.rows() {
            println!("{}", app.terminal.line(row).trim_end());
        }
        return;
    }
    let mut pty = if external {
        app.use_byte_stream();
        Some(Pty::spawn(
            &args[1..],
            app.terminal.cols(),
            app.terminal.rows(),
        ))
    } else {
        None
    };
    let mut window =
        Window::new("Oscar", 1024, 768, WindowOptions::default()).expect("open display");
    window.set_target_fps(60);
    let (tx, chars) = mpsc::sync_channel(256);
    window.set_input_callback(Box::new(TextInput(tx)));
    let mut last_pointer = (-1, -1, false);
    while window.is_open() && !app.closed {
        let mut mods = Mods::default();
        for (left, right, flag) in [
            (HostKey::LeftShift, HostKey::RightShift, Mods::SHIFT),
            (HostKey::LeftCtrl, HostKey::RightCtrl, Mods::CTRL),
            (HostKey::LeftAlt, HostKey::RightAlt, Mods::ALT),
        ] {
            if window.is_key_down(left) || window.is_key_down(right) {
                mods.0 |= flag;
            }
        }
        let mut inputs = Vec::new();
        if let Some((x, y)) = window.get_mouse_pos(MouseMode::Clamp) {
            let p = (x as i32, y as i32, window.get_mouse_down(MouseButton::Left));
            if p != last_pointer {
                inputs.push(Input::Pointer {
                    x: p.0,
                    y: p.1,
                    down: p.2,
                });
                last_pointer = p;
            }
        }
        for c in chars.try_iter().take(256) {
            if !mods.has(Mods::CTRL) {
                inputs.push(Input::Key {
                    key: Key::Character(c),
                    mods,
                });
            }
        }
        for key in window.get_keys_pressed(KeyRepeat::Yes) {
            let key = match key {
                HostKey::Enter => Some(Key::Enter),
                HostKey::Backspace => Some(Key::Backspace),
                HostKey::Delete => Some(Key::Delete),
                HostKey::Left => Some(Key::Left),
                HostKey::Right => Some(Key::Right),
                HostKey::Up => Some(Key::Up),
                HostKey::Down => Some(Key::Down),
                HostKey::Home => Some(Key::Home),
                HostKey::End => Some(Key::End),
                HostKey::Tab => Some(Key::Tab),
                HostKey::Escape => Some(Key::Escape),
                HostKey::PageUp => Some(Key::PageUp),
                HostKey::PageDown => Some(Key::PageDown),
                k if mods.has(Mods::CTRL) => ctrl_character(k).map(Key::Character),
                _ => None,
            };
            if let Some(key) = key {
                inputs.push(Input::Key { key, mods });
            }
        }
        for input in inputs {
            let out = server.input(input);
            deliver(&mut app, &mut env, out);
        }
        if let Some(pty) = pty.as_mut() {
            for bytes in pty.output.try_iter().take(16) {
                app.feed(&bytes);
            }
            let input = app.take_input();
            if !input.is_empty() && pty.input.write_all(&input).is_err() {
                app.closed = true;
            }
            if pty
                .child
                .try_wait()
                .expect("wait terminal command")
                .is_some()
            {
                app.closed = true;
            }
        }
        frame(&mut server, &mut app, &mut env);
        window
            .update_with_buffer(server.render().pixels(), 1024, 768)
            .expect("present display");
    }
    let close = app.close_packet();
    dispatch(&mut server, &mut app, &mut env, close);
}
fn ctrl_character(key: HostKey) -> Option<char> {
    // minifb reports control keys separately from its text callback.
    Some(match key {
        HostKey::A => 'a',
        HostKey::B => 'b',
        HostKey::C => 'c',
        HostKey::D => 'd',
        HostKey::E => 'e',
        HostKey::F => 'f',
        HostKey::G => 'g',
        HostKey::H => 'h',
        HostKey::I => 'i',
        HostKey::J => 'j',
        HostKey::K => 'k',
        HostKey::L => 'l',
        HostKey::M => 'm',
        HostKey::N => 'n',
        HostKey::O => 'o',
        HostKey::P => 'p',
        HostKey::Q => 'q',
        HostKey::R => 'r',
        HostKey::S => 's',
        HostKey::T => 't',
        HostKey::U => 'u',
        HostKey::V => 'v',
        HostKey::W => 'w',
        HostKey::X => 'x',
        HostKey::Y => 'y',
        HostKey::Z => 'z',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_pty_round_trip_and_ansi_output() {
        let argv = [
            "/bin/sh".into(),
            "-c".into(),
            r#"printf '\033[31mready\033[0m\n'; read -r line; printf 'reply:%s\n' "$line""#.into(),
        ];
        let mut pty = Pty::spawn(&argv, 80, 24);
        pty.input.write_all(b"oscar\n").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut bytes = Vec::new();
        while !bytes.windows(11).any(|b| b == b"reply:oscar") {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            bytes.extend(
                pty.output
                    .recv_timeout(remaining)
                    .expect("PTY reply before deadline"),
            );
        }
        assert!(bytes.windows(5).any(|b| b == b"\x1b[31m"));
        let mut terminal = oscar_desktop::terminal::Terminal::new(80, 24).unwrap();
        terminal.feed(&bytes);
        assert!((0..24).any(|row| terminal.line(row).contains("reply:oscar")));
    }
}
