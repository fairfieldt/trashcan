//! Trash command shell. Editing and command parsing are independent of filesystem
//! access; backends supply only the operations granted to this shell's session.
use crate::{
    protocol::{Key, MAX_TEXT, Mods},
    terminal::Terminal,
};
use alloc::{
    collections::VecDeque,
    format,
    string::{String, ToString},
    vec::Vec,
};

pub trait Environment {
    fn pwd(&self) -> String;
    fn cd(&mut self, path: &str) -> Result<(), String>;
    fn list(&mut self, path: &str) -> Result<String, String>;
    fn read(&mut self, path: &str) -> Result<String, String>;
}

/// Quotes and backslash escaping, with empty quoted arguments preserved. Unsupported
/// operators fail explicitly instead of accidentally invoking a different command.
pub fn parse(line: &str) -> Result<Vec<String>, &'static str> {
    if line.len() > MAX_TEXT {
        return Err("command too long");
    }
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut escape = false;
    let mut started = false;
    for c in line.chars() {
        if escape {
            word.push(c);
            escape = false;
            started = true;
            continue;
        }
        if c == '\\' && quote != Some('\'') {
            escape = true;
            started = true;
            continue;
        }
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else {
                word.push(c);
            }
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                started = true;
            }
            '|' | '&' | ';' | '<' | '>' => return Err("operators are not supported"),
            c if c.is_whitespace() => {
                if started {
                    words.push(core::mem::take(&mut word));
                    started = false;
                }
            }
            c => {
                word.push(c);
                started = true;
            }
        }
        if words.len() > 64 {
            return Err("too many arguments");
        }
    }
    if escape || quote.is_some() {
        return Err("unfinished quote or escape");
    }
    if started {
        words.push(word);
    }
    Ok(words)
}

#[derive(Default)]
pub struct Shell {
    line: String,
    cursor: usize,
    history: VecDeque<String>,
    history_at: Option<usize>,
    draft: String,
    pub exited: bool,
}
impl Shell {
    pub fn start(&self, terminal: &mut Terminal) {
        terminal.feed(b"trash% ");
    }
    pub fn line(&self) -> &str {
        &self.line
    }
    pub fn command(&mut self, line: &str, terminal: &mut Terminal, env: &mut impl Environment) {
        if line.len() > MAX_TEXT || self.exited {
            return;
        }
        self.line.clear();
        self.cursor = 0;
        self.history_at = None;
        terminal.feed(b"\r\x1b[Ktrash% ");
        terminal.feed(line.as_bytes());
        terminal.feed(b"\r\n");
        self.execute(line, terminal, env);
        if !self.exited {
            self.start(terminal);
        }
    }
    pub fn key(
        &mut self,
        key: Key,
        mods: Mods,
        terminal: &mut Terminal,
        env: &mut impl Environment,
    ) {
        if self.exited {
            return;
        }
        terminal.scroll_view(-10000);
        if mods.has(Mods::CTRL) {
            match key {
                Key::Character('c' | 'C') => {
                    self.line.clear();
                    self.cursor = 0;
                    self.history_at = None;
                    terminal.feed(b"^C\r\n");
                    self.start(terminal);
                }
                Key::Character('l' | 'L') => {
                    terminal.feed(b"\x1b[2J\x1b[H");
                    self.redraw(terminal);
                }
                Key::Character('u' | 'U') => {
                    self.line.drain(..self.cursor);
                    self.cursor = 0;
                    self.redraw(terminal);
                }
                Key::Character('a' | 'A') => {
                    self.cursor = 0;
                    self.redraw(terminal);
                }
                Key::Character('e' | 'E') => {
                    self.cursor = self.line.len();
                    self.redraw(terminal);
                }
                Key::Character('d' | 'D') if self.line.is_empty() => self.exited = true,
                _ => {}
            }
            return;
        }
        match key {
            Key::Enter => {
                let line = core::mem::take(&mut self.line);
                self.cursor = 0;
                self.history_at = None;
                terminal.feed(b"\r\x1b[Ktrash% ");
                terminal.feed(line.as_bytes());
                terminal.feed(b"\r\n");
                self.execute(&line, terminal, env);
                if !self.exited {
                    self.start(terminal);
                }
                return;
            }
            Key::Character(c) if !c.is_control() && !mods.has(Mods::ALT) => {
                if self.line.len() + c.len_utf8() <= MAX_TEXT {
                    self.line.insert(self.cursor, c);
                    self.cursor += c.len_utf8();
                }
            }
            Key::Backspace if self.cursor > 0 => {
                let prev = previous(&self.line, self.cursor);
                self.line.drain(prev..self.cursor);
                self.cursor = prev;
            }
            Key::Delete if self.cursor < self.line.len() => {
                let next = next(&self.line, self.cursor);
                self.line.drain(self.cursor..next);
            }
            Key::Left => self.cursor = previous(&self.line, self.cursor),
            Key::Right => self.cursor = next(&self.line, self.cursor),
            Key::Home => self.cursor = 0,
            Key::End => self.cursor = self.line.len(),
            Key::Up if !self.history.is_empty() => {
                if self.history_at.is_none() {
                    self.draft = self.line.clone();
                }
                let at = self
                    .history_at
                    .unwrap_or(self.history.len())
                    .saturating_sub(1);
                self.line = self.history[at].clone();
                self.history_at = Some(at);
                self.cursor = self.line.len();
            }
            Key::Down => {
                if let Some(at) = self.history_at {
                    if at + 1 < self.history.len() {
                        self.line = self.history[at + 1].clone();
                        self.history_at = Some(at + 1);
                    } else {
                        self.line = self.draft.clone();
                        self.history_at = None;
                    }
                    self.cursor = self.line.len();
                }
            }
            _ => {}
        }
        self.redraw(terminal);
    }
    fn redraw(&self, terminal: &mut Terminal) {
        let available = terminal.cols().saturating_sub(8);
        let cursor = self.line[..self.cursor].chars().count();
        let start = cursor.saturating_sub(available);
        let visible: String = self.line.chars().skip(start).take(available).collect();
        terminal.feed(b"\r\x1b[Ktrash% ");
        terminal.feed(visible.as_bytes());
        let back = visible.chars().count().saturating_sub(cursor - start);
        if back > 0 {
            terminal.feed(format!("\x1b[{back}D").as_bytes());
        }
    }
    fn execute(&mut self, line: &str, terminal: &mut Terminal, env: &mut impl Environment) {
        if line.trim().is_empty() {
            return;
        }
        if self.history.back().is_none_or(|last| last != line) {
            if self.history.len() == 64 {
                self.history.pop_front();
            }
            self.history.push_back(line.to_string());
        }
        let words = match parse(line) {
            Ok(w) => w,
            Err(e) => {
                output(terminal, &format!("trash: {e}"));
                return;
            }
        };
        if words.is_empty() {
            return;
        }
        let args = &words[1..];
        let result = match words[0].as_str() {
            "help" => Ok(String::from(
                "help  echo  clear  history  pwd  cd  ls  cat  exit\nQuotes: 'text' or \"text\"; backslash escapes a character.\nUp/Down: history. Ctrl-C: cancel. Ctrl-L: clear.",
            )),
            "echo" => Ok(args.join(" ")),
            "pwd" if args.is_empty() => Ok(env.pwd()),
            "cd" if args.len() <= 1 => env
                .cd(args.first().map_or("/", |s| s.as_str()))
                .map(|_| String::new()),
            "ls" if args.len() <= 1 => env.list(args.first().map_or(".", |s| s.as_str())),
            "cat" if args.len() == 1 => env.read(&args[0]),
            "clear" if args.is_empty() => {
                terminal.clear();
                Ok(String::new())
            }
            "history" if args.is_empty() => Ok(self
                .history
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{:>3}  {s}", i + 1))
                .collect::<Vec<_>>()
                .join("\n")),
            "exit" if args.is_empty() => {
                self.exited = true;
                Ok(String::new())
            }
            "pwd" | "cd" | "ls" | "cat" | "clear" | "history" | "exit" => {
                Err(String::from("check the command arguments"))
            }
            _ => Err(format!("{}: command not found", words[0])),
        };
        match result {
            Ok(s) if !s.is_empty() => output(terminal, &s),
            Err(e) => output(terminal, &format!("trash: {e}")),
            _ => {}
        }
    }
}
fn previous(s: &str, at: usize) -> usize {
    s[..at].char_indices().next_back().map_or(0, |(i, _)| i)
}
fn next(s: &str, at: usize) -> usize {
    s[at..].chars().next().map_or(at, |c| at + c.len_utf8())
}
fn output(t: &mut Terminal, s: &str) {
    // A backend cannot force unbounded output into one UI event.
    let mut len = s.len().min(32 * 1024);
    while !s.is_char_boundary(len) {
        len -= 1;
    }
    for line in s[..len].split('\n') {
        t.feed(line.trim_end_matches('\r').as_bytes());
        t.feed(b"\r\n");
    }
}
