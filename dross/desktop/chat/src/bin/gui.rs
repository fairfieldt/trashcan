//! oscar-chat-gui: the chat window on Oscar's desktop. A Linux program on dross: the window server on its
//! `window` channel (bootstrap), the trashcan server over TCP from a second thread, which wakes the window's loop
//! through a dross notification. Bootstrap values: `width`, `height` (the window's), `server` (HOST:PORT; else
//! as for oscar-chat).

use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_chat::{Chat, plain, server_address};
use oscar_desktop::{
    chat::{ChatWindow, HEIGHT, Request, WIDTH},
    protocol::{MAX_PACKET, Packet},
};
use std::sync::mpsc;

enum Event {
    Start,
    Text(String),
    Done,
    Failed(String),
}

fn send(ch: Handle, p: &Packet) -> Result<(), Error> {
    let bytes = p.encode().map_err(|_| Error::InvalidArgs)?;
    sys::channel_write(ch, &bytes, &[])
}

/// A failure, short enough for the status line.
fn status(e: &str) -> String {
    if e.starts_with("can't reach") {
        return String::from("Can't reach the server");
    }
    let mut s: String = e.chars().take(44).collect();
    if s.len() < e.len() {
        s.push_str("...");
    }
    s
}

fn main() {
    let b = Bootstrap::read(Handle(1)).expect("chat bootstrap");
    let _ = sys::close(Handle(1));
    let window = b.handle("window").expect("window channel");
    let width = b.value_u64("width").map_or(WIDTH, |v| v as u32);
    let height = b.value_u64("height").map_or(HEIGHT, |v| v as u32);
    let mut app = ChatWindow::new(width, height).expect("window size");
    let notify = sys::notify_create().expect("notification");
    sys::channel_bind(window, notify, 1).expect("bind window");

    let (requests, work) = mpsc::channel::<Request>();
    let (events, news) = mpsc::channel::<Event>();
    let server = server_address(b.value("server").map(String::from));
    std::thread::spawn(move || {
        let mut chat = Chat::new(server);
        let tell = |e: Event| {
            let _ = events.send(e);
            let _ = sys::notify_signal(notify, 2);
        };
        for r in work {
            match r {
                Request::Reset => chat.reset(),
                Request::Send(text) => {
                    tell(Event::Start);
                    match chat.ask(&text, |d| tell(Event::Text(plain(d)))) {
                        Ok(_) => tell(Event::Done),
                        Err(e) => tell(Event::Failed(status(&e))),
                    }
                }
            }
        }
    });

    send(window, &app.create()).expect("create window");
    let mut bytes = vec![0u8; MAX_PACKET];
    let mut handles = [Handle(0); dross_abi::MSG_HANDLES];
    loop {
        for _ in 0..16 {
            match sys::channel_read(window, &mut bytes, &mut handles, 0) {
                Ok((n, 0)) => match Packet::decode(&bytes[..n]) {
                    Ok(p) => {
                        if let Err(e) = app.receive(p) {
                            eprintln!("the window server refused the window: {e:?}");
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!("bad message from the window server: {e:?}");
                        return;
                    }
                },
                Ok((_, nh)) => {
                    for &h in &handles[..nh] {
                        let _ = sys::close(h);
                    }
                    return;
                }
                Err(Error::ShouldWait | Error::TimedOut) => break,
                Err(e) => {
                    eprintln!("lost the window server: {e:?}");
                    return;
                }
            }
        }
        for e in news.try_iter() {
            match e {
                Event::Start => app.reply_start(),
                Event::Text(t) => app.reply_text(&t),
                Event::Done => app.reply_done(),
                Event::Failed(why) => app.reply_failed(&why),
            }
        }
        for r in app.take_requests() {
            let _ = requests.send(r);
        }
        if app.closed {
            let _ = send(window, &app.close_packet());
            return;
        }
        if app.frame(|p| send(window, &p).map_err(|_| oscar_desktop::protocol::Fault::Limit)).is_err() {
            eprintln!("the window server stopped taking frames");
            return;
        }
        let _ = sys::notify_wait(notify, sys::after_ms(100));
    }
}
