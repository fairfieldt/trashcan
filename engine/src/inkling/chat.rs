//! The Inkling chat format, rendered natively (the GGUF template uses jinja features minijinja lacks): messages
//! `<|message_ROLE|><|content_text|>...<|end_message|>`, reasoning in `<|content_thinking|>`, tool calls as
//! `<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>`, the thinking effort as a
//! system message before the first other message, and `<|content_model_end_sampling|>` ending an assistant turn.
//! Generation starts right after the rendered prompt: the model writes its own role token.
use crate::tokenizer::Tokenizer;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

/// `reasoning_effort` names and their levels in the template.
pub fn effort_level(effort: Option<&Value>) -> Result<f64> {
    let Some(e) = effort else { return Ok(0.9) };
    let n = match e {
        Value::String(s) => match s.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            other => bail!("unknown reasoning_effort {other:?}"),
        },
        Value::Number(n) => n.as_f64().context("reasoning_effort")?,
        Value::Null => 0.9,
        other => bail!("reasoning_effort must be a name or a number, not {other}"),
    };
    anyhow::ensure!((0.0..=0.99).contains(&n), "reasoning_effort must be in [0.0, 0.99]");
    Ok(n)
}

fn effort_message(level: f64) -> String {
    let v = if level == 0.0 { "0".to_owned() } else { format!("{level}") };
    format!("<|message_system|><|content_text|>Thinking effort level: {v}<|end_message|>")
}

/// JSON as the template's `tojson(separators=(",", ":"))` writes it (serde_json's maps are sorted).
fn compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

/// Render `messages` (OpenAI-style: role, content (a string or parts), reasoning_content, tool_calls, name,
/// tool_call_id) with `tools` and the thinking effort. Image parts render as `<|content_image|>` plus one
/// placeholder token, which the vision path replaces by the image's embeddings.
pub fn render(messages: &[Value], tools: &[Value], effort: Option<&Value>) -> Result<String> {
    let level = effort_level(effort)?;
    let mut out = String::new();
    if !tools.is_empty() {
        let specs: Vec<Value> = tools
            .iter()
            .map(|t| {
                let f = t.get("function").unwrap_or(t);
                json!({
                    "description": f.get("description").and_then(Value::as_str).unwrap_or(""),
                    "name": f.get("name").cloned().unwrap_or(Value::Null),
                    "parameters": f.get("parameters").filter(|p| !p.is_null()).cloned().unwrap_or(json!({})),
                    "type": t.get("type").and_then(Value::as_str).unwrap_or("function"),
                })
            })
            .collect();
        out.push_str("<|message_system|>tool_declare<|content_xml|>");
        out.push_str(&compact(&Value::Array(specs)));
        out.push_str("<|end_message|>");
    }
    let mut effort_done = false;
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().context("message role")?;
        let token = match role {
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "system" => "<|message_system|>",
            "tool" => "<|message_tool|>",
            other => bail!("unknown message role {other}"),
        };
        if !effort_done && role != "system" {
            out.push_str(&effort_message(level));
            effort_done = true;
        }
        let mut rtok = token.to_owned();
        if role == "tool" {
            if let Some(n) = m.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                rtok.push_str(n);
            } else if let Some(id) = m.get("tool_call_id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                for prev in &messages[..i] {
                    for tc in prev.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                        if tc.get("id").and_then(Value::as_str) == Some(id) {
                            if let Some(n) = tc["function"]["name"].as_str() {
                                rtok.push_str(n);
                            }
                        }
                    }
                }
            }
        } else if role == "assistant" {
            if let Some(n) = m.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                rtok.push_str(n);
            }
        }
        let tool_calls = m.get("tool_calls").and_then(Value::as_array).filter(|t| !t.is_empty());
        let reasoning = m.get("reasoning_content").and_then(Value::as_str).filter(|s| !s.is_empty());
        if role == "assistant" {
            if let Some(r) = reasoning {
                out.push_str(&format!("{rtok}<|content_thinking|>{r}<|end_message|>"));
            }
        }
        match m.get("content") {
            Some(Value::String(s)) => {
                if !s.is_empty() || !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) {
                    out.push_str(&format!("{rtok}<|content_text|>{s}<|end_message|>"));
                }
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    match part {
                        Value::String(s) => out.push_str(&format!("{rtok}<|content_text|>{s}<|end_message|>")),
                        _ => {
                            let ty = part.get("type").and_then(Value::as_str).unwrap_or("text");
                            match ty {
                                "text" | "input_text" => {
                                    let t = part.get("text").and_then(Value::as_str).unwrap_or("");
                                    out.push_str(&format!("{rtok}<|content_text|>{t}<|end_message|>"));
                                }
                                "image" | "input_image" | "image_url" => {
                                    anyhow::ensure!(role == "user" || role == "tool", "images are only supported in user and tool messages");
                                    out.push_str(&format!("{rtok}<|content_image|>{IMAGE_PLACEHOLDER}<|end_message|>"));
                                }
                                other => bail!("unsupported content part type {other}"),
                            }
                        }
                    }
                }
            }
            Some(Value::Null) | None => {
                if !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) {
                    out.push_str(&format!("{rtok}<|content_text|><|end_message|>"));
                }
            }
            Some(other) => bail!("unsupported message content {other}"),
        }
        if role == "assistant" {
            for tc in tool_calls.into_iter().flatten() {
                let f = &tc["function"];
                let name = f["name"].as_str().context("tool call function name must be a string")?;
                let args = match &f["arguments"] {
                    Value::String(s) => serde_json::from_str(s).context("tool call arguments")?,
                    Value::Null => json!({}),
                    v => v.clone(),
                };
                let obj = args.as_object().context("tool call arguments must be an object")?;
                let body: Vec<String> = obj.iter().map(|(k, v)| format!("{}:{}", compact(&json!(k)), compact(v))).collect();
                out.push_str(&format!(
                    "<|message_model|>{name}<|content_invoke_tool_json|>{{\"name\":{},\"args\":{{{}}}}}<|end_message|>",
                    compact(&json!(name)),
                    body.join(",")
                ));
            }
            out.push_str("<|content_model_end_sampling|>");
        }
    }
    if !effort_done {
        out.push_str(&effort_message(level));
    }
    Ok(out)
}

/// The template's image placeholder token.
pub const IMAGE_PLACEHOLDER: &str = "<|unused_200054|>";

/// Special token ids of the format.
#[derive(Clone, Copy, Debug)]
pub struct Specials {
    pub message_model: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub invoke_tool_json: u32,
    pub end_message: u32,
    pub end_sampling: u32,
}

impl Specials {
    pub fn new(t: &Tokenizer) -> Result<Self> {
        let id = |s: &str| t.id(s).with_context(|| format!("missing special token {s}"));
        Ok(Self {
            message_model: id("<|message_model|>")?,
            content_text: id("<|content_text|>")?,
            content_thinking: id("<|content_thinking|>")?,
            invoke_tool_json: id("<|content_invoke_tool_json|>")?,
            end_message: id("<|end_message|>")?,
            end_sampling: id("<|content_model_end_sampling|>")?,
        })
    }
}

/// What the model is writing, as its tokens arrive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Between messages, or a message header (role token and name).
    Header,
    Text,
    Thinking,
    ToolCall,
}

/// Splits the model's output into thinking, text and tool calls.
pub struct Output {
    sp: Specials,
    pub part: Part,
    /// The name after `<|message_model|>` of the current message (tool calls carry the tool's name there).
    header: Vec<u8>,
    tool: Vec<u8>,
    pub text: String,
    pub thinking: String,
    pub tool_calls: Vec<(String, Value)>,
    pub done: bool,
}

/// One event of the output stream.
#[derive(Debug, PartialEq)]
pub enum Event {
    Text(String),
    Thinking(String),
    Tool(String, Value),
}

impl Output {
    pub fn new(sp: Specials) -> Self {
        Self { sp, part: Part::Header, header: Vec::new(), tool: Vec::new(), text: String::new(), thinking: String::new(), tool_calls: Vec::new(), done: false }
    }

    /// Take one generated token (`bytes`: its text) and return what it adds.
    pub fn token(&mut self, id: u32, bytes: &[u8]) -> Option<Event> {
        let sp = self.sp;
        if id == sp.end_sampling {
            self.done = true;
            return None;
        }
        if id == sp.message_model {
            self.part = Part::Header;
            self.header.clear();
            return None;
        }
        if id == sp.content_text {
            self.part = Part::Text;
            return None;
        }
        if id == sp.content_thinking {
            self.part = Part::Thinking;
            return None;
        }
        if id == sp.invoke_tool_json {
            self.part = Part::ToolCall;
            self.tool.clear();
            return None;
        }
        if id == sp.end_message {
            let ev = if self.part == Part::ToolCall {
                let header = String::from_utf8_lossy(&self.header).into_owned();
                let v: Value = serde_json::from_slice(&self.tool).unwrap_or(Value::Null);
                let name = v.get("name").and_then(Value::as_str).map(str::to_owned).unwrap_or(header);
                let args = v.get("args").cloned().unwrap_or(json!({}));
                self.tool_calls.push((name.clone(), args.clone()));
                Some(Event::Tool(name, args))
            } else {
                None
            };
            self.part = Part::Header;
            self.header.clear();
            return ev;
        }
        let s = String::from_utf8_lossy(bytes).into_owned();
        match self.part {
            Part::Header => {
                self.header.extend_from_slice(bytes);
                None
            }
            Part::Text => {
                self.text.push_str(&s);
                Some(Event::Text(s))
            }
            Part::Thinking => {
                self.thinking.push_str(&s);
                Some(Event::Thinking(s))
            }
            Part::ToolCall => {
                self.tool.extend_from_slice(bytes);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn renders_like_the_template() {
        let m = vec![json!({"role": "system", "content": "Be brief."}), json!({"role": "user", "content": "Hi"})];
        let r = render(&m, &[], Some(&json!(0.0))).unwrap();
        assert_eq!(
            r,
            "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"
        );
        let a = vec![
            json!({"role": "user", "content": "Weather?"}),
            json!({"role": "assistant", "content": "", "reasoning_content": "call it", "tool_calls": [{"id": "1", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "sunny"}),
        ];
        let r = render(&a, &[], None).unwrap();
        assert_eq!(
            r,
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Weather?<|end_message|><|message_model|><|content_thinking|>call it<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>sunny<|end_message|>"
        );
    }
    #[test]
    fn tool_declaration_is_sorted_and_compact() {
        let tools = vec![json!({"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})];
        let r = render(&[json!({"role": "user", "content": "x"})], &tools, Some(&json!(0.2))).unwrap();
        assert!(r.starts_with("<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get the current weather for a city\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"));
    }
}
