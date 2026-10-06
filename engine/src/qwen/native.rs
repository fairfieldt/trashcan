//! Qwen's native output syntax for the Responses server: reasoning between `<think>` and `</think>` (the
//! prompt opens it when thinking is on) and tool calls as
//! `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`.
use crate::{
    responses::{Output, Tool},
    tokenizer::Tokenizer,
};
use anyhow::{ensure, Context, Result};
use serde_json::{Map, Value};
use std::io::Write;

#[derive(Clone, Copy)]
pub struct Tokens {
    think: u32,
    end_think: u32,
    call: u32,
    end_call: u32,
}
impl Tokens {
    pub fn new(t: &Tokenizer) -> Result<Self> {
        let id = |s: &str| t.id(s).with_context(|| format!("vocabulary lacks {s}"));
        Ok(Self { think: id("<think>")?, end_think: id("</think>")?, call: id("<tool_call>")?, end_call: id("</tool_call>")? })
    }
}

#[derive(PartialEq)]
enum Mode {
    Text,
    Reasoning,
    Tool,
}

/// Streams visible text and reasoning without surrounding blank lines, and turns completed tool calls
/// into Responses items.
pub struct QwenOutput {
    tokens: Tokens,
    mode: Mode,
    bytes: Vec<u8>,
    /// Whitespace held back until more text follows; dropped at the end of a part.
    held: String,
    started: bool,
    pub reasoning_tokens: usize,
    pub tool_calls: usize,
}

impl QwenOutput {
    /// `thinking`: the prompt ends inside an open `<think>` block.
    pub fn new(tokens: Tokens, thinking: bool) -> Self {
        Self {
            tokens,
            mode: if thinking { Mode::Reasoning } else { Mode::Text },
            bytes: Vec::new(),
            held: String::new(),
            started: false,
            reasoning_tokens: 0,
            tool_calls: 0,
        }
    }

    pub fn token<W: Write>(&mut self, id: u32, bytes: &[u8], out: &mut Output<W>, tools: &[Tool]) -> Result<()> {
        let k = self.tokens;
        if id == k.think || id == k.end_think || id == k.call {
            ensure!(self.mode != Mode::Tool, "unexpected reasoning or tool opener inside a tool call");
            self.flush(out, true)?;
            self.mode = if id == k.think {
                Mode::Reasoning
            } else if id == k.call {
                Mode::Tool
            } else {
                Mode::Text
            };
        } else if id == k.end_call {
            ensure!(self.mode == Mode::Tool, "unexpected end of tool call");
            let raw = String::from_utf8(std::mem::take(&mut self.bytes))?;
            let (name, arguments) = parse_call(&raw, tools)?;
            out.tool(&name, arguments, tools)?;
            self.tool_calls += 1;
            self.mode = Mode::Text;
        } else {
            if self.mode == Mode::Reasoning {
                self.reasoning_tokens += 1;
            }
            self.bytes.extend_from_slice(bytes);
            self.flush(out, false)?;
        }
        Ok(())
    }

    fn flush<W: Write>(&mut self, out: &mut Output<W>, end: bool) -> Result<()> {
        if self.mode == Mode::Tool {
            return Ok(());
        }
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(_) => self.bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        let mut text = std::mem::take(&mut self.held);
        text.push_str(std::str::from_utf8(&self.bytes[..valid])?);
        self.bytes.drain(..valid);
        if !self.started {
            text = text.trim_start().to_owned();
        }
        let body = text.trim_end().len();
        if body > 0 {
            out.text(&text[..body], self.mode == Mode::Reasoning)?;
            self.started = true;
        }
        self.held = text[body..].to_owned();
        if end {
            self.held.clear();
            self.bytes.clear();
            self.started = false;
        }
        Ok(())
    }

    pub fn finish<W: Write>(&mut self, out: &mut Output<W>, limited: bool) -> Result<()> {
        ensure!(limited || self.mode != Mode::Tool, "model ended inside a tool call");
        self.flush(out, true)
    }
}

/// `<function=NAME>` and its `<parameter=P>\nVALUE\n</parameter>` blocks. Values of string-typed parameters
/// stay text; others are parsed as JSON when they are valid JSON.
fn parse_call(raw: &str, tools: &[Tool]) -> Result<(String, Value)> {
    let rest = raw.trim_start();
    let rest = rest.strip_prefix("<function=").context("tool call must start with <function=")?;
    let (name, mut rest) = rest.split_once('>').context("unterminated function name")?;
    let name = name.trim();
    let schema = tools
        .iter()
        .find(|t| t.prompt_name == name)
        .map(|t| t.schema["function"]["parameters"]["properties"].clone())
        .unwrap_or(Value::Null);
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.starts_with("</function>") {
            break;
        }
        let tail = rest.strip_prefix("<parameter=").context("expected <parameter= or </function>")?;
        let (key, tail) = tail.split_once('>').context("unterminated parameter name")?;
        let (value, tail) = tail.split_once("</parameter>").context("unterminated parameter")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let typed = if schema[key.trim()]["type"] == "string" {
            Value::String(value.to_owned())
        } else {
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
        };
        ensure!(arguments.insert(key.trim().to_owned(), typed).is_none(), "duplicate tool argument {key}");
        rest = tail;
    }
    ensure!(
        rest.trim_start().strip_prefix("</function>").is_some_and(|r| r.trim().is_empty()),
        "unexpected text after </function>"
    );
    Ok((name.to_owned(), Value::Object(arguments)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::{Defaults, Native, Request};
    use serde_json::json;

    fn request(value: Value) -> Request {
        let d = Defaults { max_tokens: 64, temperature: 0., top_k: 1, top_p: 1., seed: 1, thinking: true };
        Request::parse_native(&value, d, "qwen", Native::Qwen).unwrap()
    }
    const K: Tokens = Tokens { think: 1, end_think: 2, call: 3, end_call: 4 };

    #[test]
    fn reasoning_text_and_typed_tool_arguments() {
        let r = request(json!({"input":"Hi","reasoning":{"summary":"auto"},"tools":[{"type":"function","name":"weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"code":{"type":"string"}}}}]}));
        let mut out = Output::new(Vec::new(), false, "qwen", &r);
        let mut q = QwenOutput::new(K, true);
        for (id, text) in [
            (9, "Look up "),
            (9, "Paris.\n"),
            (2, ""),
            (9, "\n\nChecking"),
            (9, " now.\n\n"),
            (3, ""),
            (9, "\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"),
            (9, "<parameter=code>\n42\n</parameter>\n</function>\n"),
            (4, ""),
        ] {
            q.token(id, text.as_bytes(), &mut out, &r.tools).unwrap();
        }
        q.finish(&mut out, false).unwrap();
        let response = out.finish(1, 0, 9, q.reasoning_tokens, false).unwrap();
        assert_eq!(response["output"][0]["summary"][0]["text"], "Look up Paris.");
        assert_eq!(response["output"][1]["content"][0]["text"], "Checking now.");
        assert_eq!(response["output"][2]["name"], "weather");
        assert_eq!(response["output"][2]["arguments"], json!({"city":"Paris","days":3,"code":"42"}).to_string());
        assert_eq!(q.reasoning_tokens, 2);
    }

    #[test]
    fn malformed_calls_are_errors() {
        for raw in ["weather>", "<function=x>\n<parameter=a>\n1\n", "<function=x>\n<parameter=a>1</parameter><parameter=a>2</parameter></function>"] {
            assert!(parse_call(raw, &[]).is_err(), "accepted {raw}");
        }
        assert_eq!(parse_call("<function=f>\n</function>\n", &[]).unwrap(), ("f".into(), json!({})));
    }

    #[test]
    fn tool_results_become_tool_messages_in_order() {
        let r = request(json!({"input":[
            {"role":"user","content":"Weather?"},
            {"type":"function_call","call_id":"a","name":"weather","arguments":"{\"city\":\"Paris\"}"},
            {"type":"function_call_output","call_id":"a","output":"18 C"},
            {"role":"assistant","content":"It is 18 C."},
            {"role":"user","content":"Thanks"}]}));
        let roles: Vec<_> = r.messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "assistant", "user"]);
        assert_eq!(r.messages[2]["content"], "18 C");
        assert_eq!(r.messages[1]["tool_calls"][0]["function"]["arguments"], json!({"city":"Paris"}));
    }
}
