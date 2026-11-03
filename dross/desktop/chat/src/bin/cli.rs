//! oscar-chat: talk to Oscar the Grouch in a terminal.
//!
//!   oscar-chat [--server HOST:PORT]
//!
//! Type a message and press Enter. `/reset` starts a new conversation; `/quit` or Ctrl-D leaves.

use oscar_chat::{Chat, plain, server_address};
use std::io::{BufRead, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut server = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--server" => server = args.next(),
            "-h" | "--help" => {
                println!("usage: oscar-chat [--server HOST:PORT]\n/reset starts over, /quit leaves");
                return;
            }
            other => {
                eprintln!("oscar-chat: unknown argument {other} (try --help)");
                std::process::exit(2);
            }
        }
    }
    let color = std::env::var("TERM").is_ok_and(|t| !t.is_empty() && t != "dumb");
    let (you, oscar, dim, off) = if color {
        ("\x1b[1;31m", "\x1b[1;32m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    let mut chat = Chat::new(server_address(server));
    println!("{dim}Talking to Oscar at {}. /reset starts over, /quit leaves.{off}", chat.server());
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut line = String::new();
    loop {
        print!("{you}you>{off} ");
        let _ = out.flush();
        line.clear();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            return;
        }
        let text = line.trim();
        match text {
            "" => continue,
            "/quit" | "/exit" => return,
            "/reset" => {
                chat.reset();
                println!("{dim}New conversation.{off}");
                continue;
            }
            _ => {}
        }
        print!("{oscar}oscar>{off} ");
        let _ = out.flush();
        let result = chat.ask(text, |d| {
            print!("{}", plain(d));
            let _ = std::io::stdout().flush();
        });
        match result {
            Ok(_) => println!(),
            Err(e) => println!("\n{dim}({e}){off}"),
        }
    }
}
