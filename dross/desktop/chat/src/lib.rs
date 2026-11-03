//! Chat with Oscar the Grouch: a trashcan server's Responses API (`POST /v1/responses`, streamed as server-sent
//! events), with Oscar's system prompt and the conversation so far. Shared by the `oscar-chat` command and the
//! desktop's chat window.
//!
//! The server comes from `--server HOST:PORT`, else `OSCAR_CHAT_SERVER`, else a `server HOST:PORT` line in
//! `/etc/oscar-chat.conf`, else 127.0.0.1:8000. Plain HTTP: the server has no TLS.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub const SYSTEM_PROMPT: &str = "You are Oscar the Grouch from Sesame Street, chatting with someone through the \
operating system on this computer, which is named after you. You are a little grumpy and you love trash, but you \
are good-natured underneath and genuinely helpful: answer what is asked, with a grumble or two. Keep replies to two \
or three sentences, in plain text without Markdown.";

const CONFIG: &str = "/etc/oscar-chat.conf";
const DEFAULT_SERVER: &str = "127.0.0.1:8000";

/// Where the server is: the argument, the environment, the config file or the default.
pub fn server_address(arg: Option<String>) -> String {
    arg.or_else(|| std::env::var("OSCAR_CHAT_SERVER").ok().filter(|s| !s.is_empty()))
        .or_else(|| {
            std::fs::read_to_string(CONFIG).ok()?.lines().find_map(|l| {
                let (k, v) = l.trim().split_once(char::is_whitespace)?;
                (k == "server").then(|| v.trim().to_string())
            })
        })
        .unwrap_or_else(|| DEFAULT_SERVER.to_string())
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    You,
    Oscar,
}

pub struct Chat {
    server: String,
    model: Option<String>,
    history: Vec<(Role, String)>,
}

impl Chat {
    pub fn new(server: String) -> Chat {
        Chat { server, model: None, history: Vec::new() }
    }

    pub fn server(&self) -> &str {
        &self.server
    }

    /// Starts a new conversation.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Sends `text` and streams Oscar's reply to `on_text` as it arrives. The exchange joins the conversation
    /// only if the reply completes.
    pub fn ask(&mut self, text: &str, mut on_text: impl FnMut(&str)) -> Result<String, String> {
        if self.model.is_none() {
            self.model = Some(self.model_id()?);
        }
        let mut input: Vec<Value> = self
            .history
            .iter()
            .map(|(role, t)| json!({"role": if *role == Role::You { "user" } else { "assistant" }, "content": t}))
            .collect();
        input.push(json!({"role": "user", "content": text}));
        let body = json!({
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": input,
            "reasoning": {"effort": "none"},
            "temperature": 0.7,
            "max_output_tokens": 400,
            "stream": true,
        });
        let mut reply = String::new();
        let mut stream = self.request("POST", "/v1/responses", Some(&body.to_string()))?;
        let mut data = String::new();
        let mut line = String::new();
        loop {
            line.clear();
            if stream.read_line(&mut line).map_err(|e| format!("reading the reply: {e}"))? == 0 {
                return Err(String::from("the server ended the reply early"));
            }
            let l = line.trim_end_matches(['\r', '\n']);
            if let Some(d) = l.strip_prefix("data:") {
                data.push_str(d.trim_start());
                continue;
            }
            if !l.is_empty() || data.is_empty() {
                continue;
            }
            let event: Value = serde_json::from_str(&data).map_err(|e| format!("bad event from the server: {e}"))?;
            data.clear();
            match event["type"].as_str().unwrap_or("") {
                "response.output_text.delta" => {
                    if let Some(d) = event["delta"].as_str() {
                        reply.push_str(d);
                        on_text(d);
                    }
                }
                "response.completed" => break,
                "response.failed" | "response.incomplete" | "error" => {
                    let message = event["response"]["error"]["message"]
                        .as_str()
                        .or(event["message"].as_str())
                        .unwrap_or("the reply failed");
                    return Err(message.to_string());
                }
                _ => {}
            }
        }
        self.history.push((Role::You, text.to_string()));
        self.history.push((Role::Oscar, reply.clone()));
        Ok(reply)
    }

    /// The server's model (`GET /v1/models`).
    fn model_id(&self) -> Result<String, String> {
        let mut r = self.request("GET", "/v1/models", None)?;
        let mut body = String::new();
        r.read_to_string(&mut body).map_err(|e| format!("reading the model list: {e}"))?;
        let v: Value = serde_json::from_str(&body).map_err(|e| format!("bad model list: {e}"))?;
        v["data"][0]["id"].as_str().map(String::from).ok_or_else(|| String::from("the server has no model loaded"))
    }

    /// One HTTP/1.1 request; the reader yields the body (chunked transfer decoded) after a 200 status.
    fn request(&self, method: &str, path: &str, body: Option<&str>) -> Result<BufReader<Body>, String> {
        let mut s = TcpStream::connect(&self.server).map_err(|e| format!("can't reach the server at {} ({method} {path}): {e}", self.server))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(300)));
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n", self.server);
        if let Some(b) = body {
            req.push_str(&format!("Content-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n", b.len()));
        }
        req.push_str("\r\n");
        req.push_str(body.unwrap_or(""));
        s.write_all(req.as_bytes()).map_err(|e| format!("sending to the server: {e}"))?;
        let mut r = BufReader::new(s);
        let mut status = String::new();
        r.read_line(&mut status).map_err(|e| format!("reading the server's answer: {e}"))?;
        let code: u16 = status.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        let (mut chunked, mut length) = (false, None);
        loop {
            let mut h = String::new();
            if r.read_line(&mut h).map_err(|e| format!("reading the server's answer: {e}"))? == 0 {
                break;
            }
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                chunked |= k.eq_ignore_ascii_case("transfer-encoding") && v.trim().eq_ignore_ascii_case("chunked");
                if k.eq_ignore_ascii_case("content-length") {
                    length = v.trim().parse::<usize>().ok();
                }
            }
        }
        // Without chunking, a Content-Length bounds the body (the server may keep the connection open).
        let (left, sized) = match length {
            Some(n) if !chunked => (n, true),
            _ => (0, false),
        };
        let mut body = BufReader::new(Body { inner: r, chunked, sized, left, done: false });
        if code != 200 {
            let mut text = String::new();
            let _ = body.read_to_string(&mut text);
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(String::from))
                .unwrap_or_else(|| status.trim().to_string());
            return Err(format!("the server said: {message}"));
        }
        Ok(body)
    }
}

/// An HTTP response body, with chunked transfer encoding undone.
pub struct Body {
    inner: BufReader<TcpStream>,
    chunked: bool,
    /// `left` is what remains of a Content-Length body.
    sized: bool,
    left: usize,
    done: bool,
}

impl Read for Body {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.sized {
            let n = buf.len().min(self.left);
            let n = if n == 0 { 0 } else { self.inner.read(&mut buf[..n])? };
            self.left -= n;
            return Ok(n);
        }
        if !self.chunked {
            return self.inner.read(buf);
        }
        if self.done {
            return Ok(0);
        }
        if self.left == 0 {
            let mut size = String::new();
            // The CRLF after the previous chunk, then the size line.
            while size.trim().is_empty() {
                size.clear();
                if self.inner.read_line(&mut size)? == 0 {
                    self.done = true;
                    return Ok(0);
                }
            }
            let hex = size.trim().split(';').next().unwrap_or("0");
            self.left = usize::from_str_radix(hex, 16)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad chunk size"))?;
            if self.left == 0 {
                self.done = true;
                return Ok(0);
            }
        }
        let n = buf.len().min(self.left);
        let n = self.inner.read(&mut buf[..n])?;
        if n == 0 {
            self.done = true;
        }
        self.left -= n;
        Ok(n)
    }
}

/// Text for a display that has only basic Latin glyphs: typographic quotes and dashes become ASCII, anything
/// else unprintable becomes `?`.
pub fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{2018}' | '\u{2019}' | '\u{201b}' | '\u{2032}' => out.push('\''),
            '\u{201c}' | '\u{201d}' | '\u{201f}' | '\u{2033}' => out.push('"'),
            '\u{2013}' | '\u{2014}' | '\u{2212}' => out.push('-'),
            '\u{2026}' => out.push_str("..."),
            '\u{00a0}' | '\t' => out.push(' '),
            '\n' => out.push('\n'),
            c if (' '..='~').contains(&c) => out.push(c),
            _ => out.push('?'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text() {
        assert_eq!(plain("Don\u{2019}t \u{201c}cheer\u{201d}\u{2026}"), "Don't \"cheer\"...");
        assert_eq!(plain("a\u{1f5d1}b"), "a?b");
    }

    #[test]
    fn chunked_body() {
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let events = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Scram\"}\n\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"!\"}\n\n\
data: {\"type\":\"response.completed\"}\n\n";
            let (a, b) = events.split_at(30);
            let reply = format!(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\n\r\n",
                a.len(),
                b.len()
            );
            s.write_all(reply.as_bytes()).unwrap();
        });
        let mut chat = Chat::new(addr.to_string());
        chat.model = Some(String::from("test"));
        let mut seen = String::new();
        assert_eq!(chat.ask("hi", |d| seen.push_str(d)).unwrap(), "Scram!");
        assert_eq!(seen, "Scram!");
        assert_eq!(chat.history.len(), 2);
    }
}
