//! Responses wire protocol and Gemma's native tool/channel syntax, independent of the GPU.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy)]
pub struct Defaults {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub thinking: bool,
}

/// Where a model's chat template expects tool results: Gemma attaches them to the calling assistant
/// message; Qwen renders them as `tool` messages after it.
#[derive(Clone, Copy, PartialEq)]
pub enum Native {
    Gemma,
    Qwen,
}

#[derive(Debug)]
pub struct Tool {
    pub name: String,
    pub namespace: Option<String>,
    pub prompt_name: String,
    pub custom: bool,
    pub search: bool,
    pub schema: Value,
}

pub struct Request {
    pub messages: Vec<Value>,
    pub tools: Vec<Tool>,
    wire_tools: Value,
    tool_choice: String,
    pub parallel_tool_calls: bool,
    pub stream: bool,
    pub thinking: bool,
    /// The requested reasoning effort, when given.
    pub effort: Option<String>,
    pub summary: bool,
    pub sampling: Defaults,
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .with_context(|| format!("{key} must be a nonempty string"))
}

fn content(v: &Value) -> Result<String> {
    if let Some(s) = v.as_str() {
        return Ok(s.to_owned());
    }
    let parts = v
        .as_array()
        .context("content must be a string or a text content array")?;
    let mut text = String::new();
    for part in parts {
        ensure!(
            matches!(
                part["type"].as_str(),
                Some("input_text" | "output_text" | "text" | "summary_text")
            ),
            "only text content is supported (images, audio and files are unsupported)"
        );
        text.push_str(
            part["text"]
                .as_str()
                .context("text content requires a text string")?,
        );
    }
    Ok(text)
}

fn qualified(name: &str, namespace: Option<&str>) -> String {
    namespace
        .map(|n| format!("{n}.{name}"))
        .unwrap_or_else(|| name.to_owned())
}

fn response_tools(mut tools: Value) -> Value {
    // A namespace description is optional for our callers, but required on SDK response objects.
    for tool in tools.as_array_mut().unwrap() {
        if tool["type"] == "namespace" && tool.get("description").is_none_or(Value::is_null) {
            tool["description"] = "".into();
        }
    }
    tools
}

fn parse_tools(v: &Value, namespace: Option<&str>, out: &mut Vec<Tool>) -> Result<()> {
    for tool in v.as_array().context("tools must be an array")? {
        ensure!(
            tool.get("description")
                .is_none_or(|v| v.is_null() || v.is_string()),
            "tool description must be a string"
        );
        let kind = string(tool, "type")?;
        if kind == "namespace" {
            ensure!(
                namespace.is_none(),
                "nested tool namespaces are unsupported"
            );
            parse_tools(&tool["tools"], Some(string(tool, "name")?), out)?;
            continue;
        }
        ensure!(
            matches!(kind, "function" | "custom" | "tool_search"),
            "unsupported tool type: {kind}"
        );
        let search = kind == "tool_search";
        if search {
            ensure!(
                namespace.is_none() && tool["execution"] == "client",
                "only client-executed tool_search is supported"
            );
        }
        let name = if search {
            "tool_search"
        } else {
            string(tool, "name")?
        };
        let prompt_name = qualified(name, namespace);
        ensure!(
            prompt_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')),
            "tool names must contain only letters, numbers, underscores, hyphens and dots"
        );
        ensure!(
            !out.iter().any(|t| t.prompt_name == prompt_name),
            "duplicate tool name: {prompt_name}"
        );
        let custom = kind == "custom";
        let mut description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let parameters = if custom {
            description.push_str("\nPass the tool's entire free-form input as the input string.");
            json!({"type":"object", "properties":{"input":{"type":"string"}}, "required":["input"]})
        } else {
            let parameters = tool
                .get("parameters")
                .filter(|v| !v.is_null())
                .cloned()
                .unwrap_or_else(|| json!({"type":"object", "properties":{}}));
            ensure!(
                parameters.is_object() && parameters["type"] == "object",
                "function parameters must be an object schema"
            );
            parameters
        };
        out.push(Tool {
            name: name.to_owned(), namespace: namespace.map(str::to_owned), prompt_name: prompt_name.clone(), custom, search,
            schema: json!({"type":"function", "function":{"name":prompt_name,"description":description,"parameters":parameters}}),
        });
    }
    Ok(())
}

impl Request {
    pub fn parse_native(v: &Value, defaults: Defaults, model: &str, native: Native) -> Result<Self> {
        ensure!(v.is_object(), "request must be a JSON object");
        if let Some(m) = v.get("model") {
            ensure!(m.as_str() == Some(model), "unknown model; use {model}");
        }
        for field in ["previous_response_id", "conversation"] {
            ensure!(
                v.get(field).is_none_or(Value::is_null),
                "{field} is unsupported; replay the full input history instead"
            );
        }
        for field in ["background", "store"] {
            ensure!(
                v.get(field).is_none_or(|v| v.is_null() || v == false),
                "{field}=true is unsupported"
            );
        }
        ensure!(
            v.get("truncation")
                .is_none_or(|v| v.is_null() || v == "disabled"),
            "automatic truncation is unsupported"
        );
        ensure!(
            v.pointer("/text/format/type").is_none_or(|v| v == "text"),
            "structured output constraints are unsupported"
        );
        let mut sampling = defaults;
        if let Some(value) = v
            .get("max_output_tokens")
            .or_else(|| v.get("max_tokens"))
            .filter(|v| !v.is_null())
        {
            sampling.max_tokens = usize::try_from(
                value
                    .as_u64()
                    .context("max_output_tokens must be a positive integer")?,
            )?;
        }
        ensure!(
            sampling.max_tokens > 0,
            "max_output_tokens must be positive"
        );
        if let Some(value) = v.get("temperature").filter(|v| !v.is_null()) {
            sampling.temperature = value.as_f64().context("temperature must be a number")? as f32;
        }
        if let Some(value) = v.get("top_p").filter(|v| !v.is_null()) {
            sampling.top_p = value.as_f64().context("top_p must be a number")? as f32;
        }
        if let Some(value) = v.get("top_k") {
            sampling.top_k = usize::try_from(value.as_u64().context("top_k must be an integer")?)?;
        }
        if let Some(value) = v.get("seed") {
            sampling.seed = value
                .as_u64()
                .context("seed must be a nonnegative integer")?;
        }
        crate::generation::Sampler::new(
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.seed,
        )?;
        let stream = v
            .get("stream")
            .map(|v| v.as_bool().context("stream must be a boolean"))
            .transpose()?
            .unwrap_or(false);
        let mut thinking = defaults.thinking;
        let mut effort_name = None;
        let mut summary = false;
        if let Some(reasoning) = v.get("reasoning").filter(|v| !v.is_null()) {
            ensure!(reasoning.is_object(), "reasoning must be an object");
            if let Some(effort) = reasoning.get("effort").filter(|v| !v.is_null()) {
                let effort = effort
                    .as_str()
                    .context("reasoning effort must be a string")?;
                ensure!(
                    matches!(
                        effort,
                        "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
                    ),
                    "unknown reasoning effort"
                );
                thinking = effort != "none";
                effort_name = Some(effort.to_owned());
            }
            if let Some(value) = reasoning.get("summary").filter(|v| !v.is_null()) {
                ensure!(
                    matches!(
                        value.as_str(),
                        Some("none" | "auto" | "concise" | "detailed")
                    ),
                    "unknown reasoning summary mode"
                );
                summary = value != "none";
            }
        }
        let mut tools = Vec::new();
        let mut wire_tools = v
            .get("tools")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or_else(|| json!([]));
        if let Some(value) = v.get("tools").filter(|v| !v.is_null()) {
            parse_tools(value, None, &mut tools)?;
        }
        let mut tool_choice = "auto".to_owned();
        if let Some(choice) = v.get("tool_choice").filter(|v| !v.is_null()) {
            ensure!(
                choice == "auto" || choice == "none",
                "forced or required tool_choice is unsupported"
            );
            if choice == "none" {
                tools.clear();
            }
            tool_choice = choice.as_str().unwrap().to_owned();
        }
        let parallel_tool_calls = v
            .get("parallel_tool_calls")
            .map(|v| v.as_bool().context("parallel_tool_calls must be a boolean"))
            .transpose()?
            .unwrap_or(true);
        let input = v.get("input").context("missing input")?;
        let mut messages: Vec<Value> = Vec::new();
        let mut calls: HashMap<String, (usize, String)> = HashMap::new();
        let mut seen_calls = HashSet::new();
        let mut pending_reasoning = String::new();
        let mut system = Vec::new();
        if let Some(instructions) = v.get("instructions").filter(|v| !v.is_null()) {
            system.push(
                instructions
                    .as_str()
                    .context("instructions must be a string")?
                    .to_owned(),
            );
        }
        if let Some(text) = input.as_str() {
            messages.push(json!({"role":"user","content":text}));
        } else {
            for item in input
                .as_array()
                .context("input must be a string or an array")?
            {
                ensure!(item.is_object(), "input items must be objects");
                if let Some(status) = item.get("status") {
                    ensure!(
                        status == "completed",
                        "replayed input items must have completed status"
                    );
                }
                match item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message")
                {
                    "message" => {
                        let role = string(item, "role")?;
                        ensure!(
                            matches!(role, "system" | "developer" | "user" | "assistant"),
                            "unsupported message role: {role}"
                        );
                        let text = content(&item["content"])?;
                        if role == "system" || role == "developer" {
                            system.push(text);
                        } else if role == "assistant"
                            && native == Native::Gemma
                            && messages
                                .last()
                                .is_some_and(|m| m.get("tool_responses").is_some())
                            && pending_reasoning.is_empty()
                        {
                            let last = messages.last_mut().unwrap();
                            let joined = last["content"].as_str().unwrap_or("").to_owned() + &text;
                            last["content"] = joined.into();
                        } else {
                            let mut message = json!({"role":role,"content":text});
                            if role == "assistant" && !pending_reasoning.is_empty() {
                                message["reasoning_content"] =
                                    std::mem::take(&mut pending_reasoning).into();
                            }
                            messages.push(message);
                        }
                    }
                    "reasoning" => {
                        let text = item
                            .get("summary")
                            .map(content)
                            .transpose()?
                            .unwrap_or_default();
                        ensure!(!text.is_empty() || item.get("encrypted_content").is_none_or(Value::is_null),
                            "encrypted reasoning cannot be replayed; use visible reasoning summaries");
                        pending_reasoning.push_str(&text);
                    }
                    kind @ ("function_call" | "custom_tool_call" | "tool_search_call") => {
                        let id = string(item, "call_id")?;
                        ensure!(seen_calls.insert(id.to_owned()), "duplicate call_id: {id}");
                        let name = if kind == "tool_search_call" {
                            "tool_search".to_owned()
                        } else {
                            qualified(
                                string(item, "name")?,
                                item.get("namespace").and_then(Value::as_str),
                            )
                        };
                        let args = if kind == "custom_tool_call" {
                            json!({"input":item["input"].as_str().context("custom tool input must be a string")?})
                        } else if kind == "tool_search_call" {
                            ensure!(item["execution"] == "client" && item["arguments"].is_object(), "tool search history requires client execution and an argument object");
                            item["arguments"].clone()
                        } else {
                            let args: Value = serde_json::from_str(
                                item["arguments"]
                                    .as_str()
                                    .context("function arguments must be a JSON string")?,
                            )?;
                            ensure!(args.is_object(), "function arguments must encode an object");
                            args
                        };
                        if messages.last().is_none_or(|m| {
                            m["role"] != "assistant" || m.get("tool_responses").is_some()
                        }) {
                            messages.push(json!({"role":"assistant","content":""}));
                        }
                        let index = messages.len() - 1;
                        let message = &mut messages[index];
                        if !pending_reasoning.is_empty() {
                            message["reasoning_content"] =
                                std::mem::take(&mut pending_reasoning).into();
                        }
                        if message.get("tool_calls").is_none() {
                            message["tool_calls"] = json!([]);
                        }
                        message["tool_calls"].as_array_mut().unwrap().push(json!({"id":id,"type":"function","function":{"name":name,"arguments":args}}));
                        calls.insert(id.to_owned(), (index, name));
                    }
                    kind @ ("function_call_output"
                    | "custom_tool_call_output"
                    | "tool_search_output") => {
                        let id = string(item, "call_id")?;
                        let (index, name) = calls
                            .remove(id)
                            .with_context(|| format!("tool output has no preceding call: {id}"))?;
                        ensure!(
                            messages[index + 1..].iter().all(|m| m["role"] == "tool"),
                            "tool outputs must follow their assistant call turn"
                        );
                        let text = if kind == "tool_search_output" {
                            ensure!(
                                item["execution"] == "client",
                                "only client tool search results can be replayed"
                            );
                            let mut loaded = Vec::new();
                            parse_tools(&item["tools"], None, &mut loaded)?;
                            for tool in loaded {
                                if !tools.iter().any(|t| t.prompt_name == tool.prompt_name) {
                                    tools.push(tool);
                                }
                            }
                            wire_tools
                                .as_array_mut()
                                .unwrap()
                                .extend(item["tools"].as_array().unwrap().iter().cloned());
                            json!({"tools":item["tools"]}).to_string()
                        } else {
                            content(&item["output"])?
                        };
                        if native == Native::Qwen {
                            messages.push(json!({"role":"tool","content":text}));
                            continue;
                        }
                        // Older embedded Gemma templates require results on the calling assistant.
                        let message = &mut messages[index];
                        if message.get("tool_responses").is_none() {
                            message["tool_responses"] = json!([]);
                        }
                        let response =
                            serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text));
                        message["tool_responses"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!({"name":name,"response":response}));
                    }
                    kind => bail!("unsupported input item type: {kind}"),
                }
            }
        }
        ensure!(
            calls.is_empty(),
            "input contains tool calls without matching outputs"
        );
        if tool_choice == "none" {
            tools.clear();
        }
        ensure!(!messages.is_empty(), "input must contain a conversation");
        if !system.is_empty() {
            messages.insert(0, json!({"role":"system","content":system.join("\n\n")}));
        }
        Ok(Self {
            messages,
            tools,
            wire_tools: response_tools(wire_tools),
            tool_choice,
            parallel_tool_calls,
            stream,
            thinking,
            effort: effort_name,
            summary,
            sampling,
        })
    }
}

static SERIAL: AtomicU64 = AtomicU64::new(0);
fn id(prefix: &str) -> String {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "{prefix}_{stamp:x}{:x}",
        SERIAL.fetch_add(1, Ordering::Relaxed)
    )
}

/// Both HTTP modes use this accumulator, keeping terminal objects identical to SSE items.
pub struct Output<W> {
    writer: W,
    streaming: bool,
    sequence: u64,
    response: Value,
    active: Option<usize>,
    summary: bool,
}

impl<W: Write> Output<W> {
    pub fn new(writer: W, streaming: bool, model: &str, request: &Request) -> Self {
        Self {
            writer,
            streaming,
            sequence: 0,
            active: None,
            summary: request.summary,
            response: json!({"id":id("resp"),"object":"response","created_at":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                "status":"in_progress","error":null,"incomplete_details":null,"model":model,"output":[],"usage":null,
                "store":false,"parallel_tool_calls":request.parallel_tool_calls,"tool_choice":request.tool_choice,"tools":request.wire_tools,"temperature":request.sampling.temperature,
                "top_p":request.sampling.top_p,"max_output_tokens":request.sampling.max_tokens}),
        }
    }
    fn event(&mut self, kind: &str, mut data: Value) -> Result<()> {
        if self.streaming {
            data["type"] = kind.into();
            data["sequence_number"] = self.sequence.into();
            self.sequence += 1;
            // Explicit chunk framing also permits persistent HTTP connections and SDK readers.
            let body = format!("event: {kind}\ndata: {data}\n\n");
            write!(self.writer, "{:x}\r\n{}\r\n", body.len(), body)?;
            self.writer.flush()?;
        }
        Ok(())
    }
    pub fn start(&mut self) -> Result<()> {
        self.event("response.created", json!({"response":self.response}))?;
        self.event("response.in_progress", json!({"response":self.response}))
    }
    fn items(&mut self) -> &mut Vec<Value> {
        self.response["output"].as_array_mut().unwrap()
    }
    fn close(&mut self, status: &str) -> Result<()> {
        let Some(index) = self.active.take() else {
            return Ok(());
        };
        self.items()[index]["status"] = status.into();
        let item = self.items()[index].clone();
        let reasoning = item["type"] == "reasoning";
        let part = if reasoning {
            &item["summary"][0]
        } else {
            &item["content"][0]
        };
        let base = json!({"item_id":item["id"],"output_index":index,"content_index":0,"summary_index":0,"text":part["text"],"logprobs":[]});
        self.event(
            if reasoning {
                "response.reasoning_summary_text.done"
            } else {
                "response.output_text.done"
            },
            base,
        )?;
        self.event(if reasoning { "response.reasoning_summary_part.done" } else { "response.content_part.done" },
            json!({"item_id":item["id"],"output_index":index,"content_index":0,"summary_index":0,"part":part}))?;
        self.event(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        )
    }
    pub fn text(&mut self, delta: &str, reasoning: bool) -> Result<()> {
        if delta.is_empty() || (reasoning && !self.summary) {
            return Ok(());
        }
        let kind = if reasoning { "reasoning" } else { "message" };
        if self
            .active
            .is_some_and(|i| self.response["output"][i]["type"] != kind)
        {
            self.close("completed")?;
        }
        let index = if let Some(i) = self.active {
            i
        } else {
            let index = self.items().len();
            let item = if reasoning {
                json!({"id":id("rs"),"type":"reasoning","status":"in_progress","summary":[]})
            } else {
                json!({"id":id("msg"),"type":"message","role":"assistant","status":"in_progress","content":[]})
            };
            self.items().push(item.clone());
            self.event(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            )?;
            let part = if reasoning {
                json!({"type":"summary_text","text":""})
            } else {
                json!({"type":"output_text","text":"","annotations":[],"logprobs":[]})
            };
            self.items()[index][if reasoning { "summary" } else { "content" }] = json!([part]);
            self.event(if reasoning { "response.reasoning_summary_part.added" } else { "response.content_part.added" },
                json!({"item_id":item["id"],"output_index":index,"content_index":0,"summary_index":0,"part":part}))?;
            self.active = Some(index);
            index
        };
        let field = if reasoning { "summary" } else { "content" };
        let text = self.items()[index][field][0]["text"]
            .as_str()
            .unwrap()
            .to_owned()
            + delta;
        self.items()[index][field][0]["text"] = text.into();
        let item_id = self.items()[index]["id"].clone();
        self.event(if reasoning { "response.reasoning_summary_text.delta" } else { "response.output_text.delta" },
            json!({"item_id":item_id,"output_index":index,"content_index":0,"summary_index":0,"delta":delta,"logprobs":[]}))
    }
    pub fn tool(&mut self, name: &str, arguments: Value, tools: &[Tool]) -> Result<()> {
        self.close("completed")?;
        let tool = tools
            .iter()
            .find(|t| t.prompt_name == name)
            .with_context(|| format!("model called an undeclared tool: {name}"))?;
        if tool.search {
            let index = self.items().len();
            let mut item = json!({"id":id("ts"),"type":"tool_search_call","status":"in_progress",
                "call_id":id("call"),"execution":"client","arguments":{}});
            self.event(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            )?;
            item["arguments"] = arguments;
            item["status"] = "completed".into();
            self.event(
                "response.output_item.done",
                json!({"output_index":index,"item":item}),
            )?;
            self.items().push(item);
            return Ok(());
        }
        let custom = tool.custom;
        let body = if custom {
            arguments["input"]
                .as_str()
                .context("custom tool call requires an input string")?
                .to_owned()
        } else {
            arguments.to_string()
        };
        let mut item = json!({"id":id(if custom {"ct"} else {"fc"}),"type":if custom {"custom_tool_call"} else {"function_call"},
            "status":"in_progress","call_id":id("call"),"name":tool.name});
        if let Some(namespace) = &tool.namespace {
            item["namespace"] = namespace.clone().into();
        }
        let field = if custom { "input" } else { "arguments" };
        item[field] = "".into();
        let index = self.items().len();
        self.event(
            "response.output_item.added",
            json!({"output_index":index,"item":item}),
        )?;
        let prefix = if custom {
            "response.custom_tool_call_input"
        } else {
            "response.function_call_arguments"
        };
        self.event(
            &format!("{prefix}.delta"),
            json!({"item_id":item["id"],"output_index":index,"delta":body}),
        )?;
        item[field] = body.clone().into();
        item["status"] = "completed".into();
        let mut done = json!({"item_id":item["id"],"output_index":index,"name":tool.name});
        done[field] = body.into();
        self.event(&format!("{prefix}.done"), done)?;
        self.event(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        )?;
        self.items().push(item);
        Ok(())
    }
    pub fn finish(
        &mut self,
        input_tokens: usize,
        cached_tokens: usize,
        output_tokens: usize,
        reasoning_tokens: usize,
        limited: bool,
    ) -> Result<Value> {
        self.close(if limited { "incomplete" } else { "completed" })?;
        self.response["status"] = if limited { "incomplete" } else { "completed" }.into();
        if limited {
            self.response["incomplete_details"] = json!({"reason":"max_output_tokens"});
        }
        self.response["usage"] = json!({"input_tokens":input_tokens,"input_tokens_details":{"cached_tokens":cached_tokens,"cache_write_tokens":0},
            "output_tokens":output_tokens,"output_tokens_details":{"reasoning_tokens":reasoning_tokens},"total_tokens":input_tokens+output_tokens});
        self.event(
            if limited {
                "response.incomplete"
            } else {
                "response.completed"
            },
            json!({"response":self.response}),
        )?;
        self.end()?;
        Ok(self.response.clone())
    }
    pub fn fail(&mut self, message: &str) -> Result<()> {
        self.close("incomplete")?;
        self.response["status"] = "failed".into();
        self.response["error"] = json!({"code":"server_error","message":message});
        self.event("response.failed", json!({"response":self.response}))?;
        self.end()
    }
    fn end(&mut self) -> Result<()> {
        if self.streaming {
            self.writer.write_all(b"0\r\n\r\n")?;
            self.writer.flush()?;
        }
        Ok(())
    }
}

#[derive(Default, PartialEq)]
enum Mode {
    #[default]
    Text,
    Reasoning,
    Tool,
}

/// Consume token bytes without leaking channel labels or native tool syntax into visible text.
#[derive(Default)]
pub struct NativeOutput {
    mode: Mode,
    bytes: Vec<u8>,
    heading: bool,
    pub reasoning_tokens: usize,
    pub tool_calls: usize,
}
impl NativeOutput {
    pub fn token<W: Write>(
        &mut self,
        id: u32,
        bytes: &[u8],
        out: &mut Output<W>,
        tools: &[Tool],
    ) -> Result<()> {
        match id {
            100 | 101 | 48 => {
                ensure!(
                    self.mode != Mode::Tool,
                    "unexpected channel or tool opener inside a tool call"
                );
                self.flush(out, true)?;
                self.mode = match id {
                    100 => Mode::Reasoning,
                    48 => Mode::Tool,
                    _ => Mode::Text,
                };
                self.heading = id == 100;
            }
            49 => {
                ensure!(self.mode == Mode::Tool, "unexpected end of tool call");
                let raw = std::str::from_utf8(&self.bytes)?.trim();
                let raw = raw
                    .strip_prefix("call:")
                    .context("tool call must start with call:")?;
                let (name, tail) = raw
                    .split_once('{')
                    .context("tool call has no argument object")?;
                let arguments = parse_arguments(&format!("{{{tail}"))?;
                out.tool(name.trim(), arguments, tools)?;
                self.tool_calls += 1;
                self.bytes.clear();
                self.mode = Mode::Text;
            }
            _ => {
                if self.mode == Mode::Reasoning {
                    self.reasoning_tokens += 1;
                }
                self.bytes.extend_from_slice(bytes);
                if self.mode != Mode::Tool {
                    self.flush(out, false)?;
                }
            }
        }
        Ok(())
    }
    fn flush<W: Write>(&mut self, out: &mut Output<W>, final_part: bool) -> Result<()> {
        if self.mode == Mode::Tool {
            return Ok(());
        }
        if self.heading {
            if let Some(nl) = self.bytes.iter().position(|b| *b == b'\n') {
                self.bytes.drain(..=nl);
                self.heading = false;
            } else {
                if final_part {
                    self.bytes.clear();
                    self.heading = false;
                }
                return Ok(());
            }
        }
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(_) => self.bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            out.text(
                std::str::from_utf8(&self.bytes[..valid])?,
                self.mode == Mode::Reasoning,
            )?;
            self.bytes.drain(..valid);
        }
        if final_part && !self.bytes.is_empty() {
            out.text(
                &String::from_utf8_lossy(&self.bytes),
                self.mode == Mode::Reasoning,
            )?;
            self.bytes.clear();
        }
        Ok(())
    }
    pub fn finish<W: Write>(&mut self, out: &mut Output<W>, limited: bool) -> Result<()> {
        ensure!(
            limited || self.mode != Mode::Tool,
            "model ended inside a tool call"
        );
        self.flush(out, true)
    }
}

// Gemma uses unquoted object keys and <|"|> string delimiters, including nested values.
fn parse_arguments(raw: &str) -> Result<Value> {
    struct Parser<'a> {
        rest: &'a str,
    }
    impl Parser<'_> {
        fn trim(&mut self) {
            self.rest = self.rest.trim_start();
        }
        fn take(&mut self, prefix: &str) -> bool {
            self.trim();
            if let Some(rest) = self.rest.strip_prefix(prefix) {
                self.rest = rest;
                true
            } else {
                false
            }
        }
        fn value(&mut self, depth: usize) -> Result<Value> {
            ensure!(depth < 64, "tool arguments are nested too deeply");
            self.trim();
            if self.take("<|\"|>") {
                let (text, rest) = self
                    .rest
                    .split_once("<|\"|>")
                    .context("unterminated tool string")?;
                self.rest = rest;
                return Ok(text.into());
            }
            if self.take("{") {
                let mut object = serde_json::Map::new();
                if self.take("}") {
                    return Ok(object.into());
                }
                loop {
                    self.trim();
                    let key = if self.rest.starts_with("<|\"|>") || self.rest.starts_with('"') {
                        self.value(depth + 1)?
                            .as_str()
                            .context("invalid tool object key")?
                            .to_owned()
                    } else {
                        let (key, rest) = self
                            .rest
                            .split_once(':')
                            .context("tool key missing colon")?;
                        ensure!(
                            !key.trim().is_empty() && !key.contains([',', '{', '}', '[', ']']),
                            "invalid tool object key"
                        );
                        self.rest = rest;
                        let key = key.trim().to_owned();
                        ensure!(
                            object.insert(key, self.value(depth + 1)?).is_none(),
                            "duplicate tool argument key"
                        );
                        if self.take("}") {
                            break;
                        }
                        ensure!(self.take(","), "tool object missing comma");
                        continue;
                    };
                    ensure!(self.take(":"), "tool key missing colon");
                    ensure!(
                        object.insert(key, self.value(depth + 1)?).is_none(),
                        "duplicate tool argument key"
                    );
                    if self.take("}") {
                        break;
                    }
                    ensure!(self.take(","), "tool object missing comma");
                }
                return Ok(object.into());
            }
            if self.take("[") {
                let mut array = Vec::new();
                if self.take("]") {
                    return Ok(array.into());
                }
                loop {
                    array.push(self.value(depth + 1)?);
                    if self.take("]") {
                        break;
                    }
                    ensure!(self.take(","), "tool array missing comma");
                }
                return Ok(array.into());
            }
            let mut stream = serde_json::Deserializer::from_str(self.rest).into_iter::<Value>();
            let value = stream.next().context("missing tool argument value")??;
            self.rest = &self.rest[stream.byte_offset()..];
            Ok(value)
        }
    }
    let mut parser = Parser { rest: raw };
    let value = parser.value(0)?;
    ensure!(
        parser.rest.trim().is_empty() && value.is_object(),
        "invalid tool argument object"
    );
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Defaults {
        Defaults {
            max_tokens: 256,
            temperature: 1.,
            top_k: 64,
            top_p: 0.95,
            seed: 1,
            thinking: false,
        }
    }
    fn request(value: Value) -> Result<Request> {
        Request::parse_native(&value, defaults(), "gemma-4-12b", Native::Gemma)
    }

    #[test]
    fn messages_sampling_and_reasoning() {
        let r = request(json!({"instructions":"Be brief", "input":[
            {"role":"developer","content":"Use metric units"},
            {"role":"user","content":[{"type":"input_text","text":"Hi"},{"type":"input_text","text":"!"}]}],
            "stream":true,"temperature":0,"top_p":1,"top_k":4,"seed":9,"max_output_tokens":42,
            "reasoning":{"effort":"high","summary":"auto"}})).unwrap();
        assert_eq!(r.messages[0]["content"], "Be brief\n\nUse metric units");
        assert_eq!(r.messages[1]["content"], "Hi!");
        assert!(r.stream && r.thinking && r.summary);
        assert_eq!(r.sampling.max_tokens, 42);
        assert_eq!(r.sampling.temperature, 0.);
        assert_eq!(r.sampling.top_k, 4);
        assert_eq!(r.sampling.seed, 9);
    }

    #[test]
    fn rejects_unsupported_and_invalid_requests() {
        for patch in [
            json!({"previous_response_id":"resp_x"}),
            json!({"conversation":"conv_x"}),
            json!({"store":true}),
            json!({"background":true}),
            json!({"truncation":"auto"}),
            json!({"tool_choice":"required"}),
            json!({"tool_choice":{"type":"function","name":"x"}}),
            json!({"temperature":-1}),
            json!({"top_p":0}),
            json!({"top_k":65}),
            json!({"stream":"true"}),
            json!({"max_output_tokens":0}),
            json!({"max_output_tokens":-1}),
            json!({"model":"other"}),
            json!({"reasoning":{"effort":"banana"}}),
            json!({"text":{"format":{"type":"json_schema"}}}),
            json!({"tools":[{"type":"web_search"}]}),
        ] {
            let mut value = json!({"input":"Hello"});
            value
                .as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            assert!(request(value.clone()).is_err(), "accepted {value}");
        }
        for input in [
            json!([]),
            json!(123),
            json!([{"role":"user","content":[{"type":"input_image","image_url":"data:..."}]}]),
            json!([{"type":"function_call_output","call_id":"missing","output":"x"}]),
            json!([{"type":"function_call","call_id":"a","name":"x","arguments":"{}"}]),
            json!([{"type":"message","role":"assistant","status":"incomplete","content":"partial"}]),
            json!([{"type":"reasoning","encrypted_content":"opaque"},{"role":"user","content":"Hi"}]),
        ] {
            assert!(request(json!({"input":input})).is_err());
        }
        assert!(
            request(json!({"input":"Hello","store":false,"previous_response_id":null})).is_ok()
        );
    }

    #[test]
    fn tool_replay_preserves_multiple_rounds_and_out_of_order_results() {
        let r = request(json!({"input":[
            {"role":"user","content":"Find the answer"},
            {"type":"reasoning","summary":[{"type":"summary_text","text":"Look up both facts"}]},
            {"type":"function_call","call_id":"a","name":"first","namespace":"tools","arguments":"{\"q\":\"one\"}"},
            {"type":"function_call","call_id":"b","name":"second","arguments":"{}"},
            {"type":"function_call_output","call_id":"b","output":"second result"},
            {"type":"function_call_output","call_id":"a","output":"{\"fact\":1}"},
            {"type":"custom_tool_call","call_id":"c","name":"patch","input":"line1\nline2"},
            {"type":"custom_tool_call_output","call_id":"c","output":[{"type":"input_text","text":"applied"}]},
            {"role":"assistant","content":[{"type":"output_text","text":"Done"}]}]})).unwrap();
        assert_eq!(r.messages.len(), 3);
        assert_eq!(r.messages[1]["reasoning_content"], "Look up both facts");
        assert_eq!(
            r.messages[1]["tool_calls"][0]["function"]["name"],
            "tools.first"
        );
        assert_eq!(r.messages[1]["tool_responses"][0]["name"], "second");
        assert_eq!(
            r.messages[1]["tool_responses"][1]["response"],
            json!({"fact":1})
        );
        assert_eq!(
            r.messages[2]["tool_calls"][0]["function"]["arguments"],
            json!({"input":"line1\nline2"})
        );
        assert_eq!(r.messages[2]["content"], "Done");
    }

    #[test]
    fn native_nested_arguments_and_strings() {
        assert_eq!(parse_arguments("{city:<|\"|>New York<|\"|>,data:{flag:true,arr:[null,-1.25e2,<|\"|>quotes \" and \\ and\nnewlines<|\"|>]},<|\"|>other<|\"|>:{}} ").unwrap(),
            json!({"city":"New York","data":{"flag":true,"arr":[null,-125.0,"quotes \" and \\ and\nnewlines"]},"other":{}}));
        assert_eq!(
            parse_arguments("{\"key\":\"JSON string\"}").unwrap(),
            json!({"key":"JSON string"})
        );
        for raw in [
            "{x:}",
            "{x:1,x:2}",
            "{x:[1,]}",
            "{x:<|\"|>unfinished}",
            "{x:1} trailing",
            "[]",
        ] {
            assert!(parse_arguments(raw).is_err(), "accepted {raw}");
        }
    }

    fn events(bytes: &[u8]) -> Vec<Value> {
        let mut remaining = bytes;
        let mut payload = Vec::new();
        loop {
            let nl = remaining.windows(2).position(|b| b == b"\r\n").unwrap();
            let length =
                usize::from_str_radix(std::str::from_utf8(&remaining[..nl]).unwrap(), 16).unwrap();
            remaining = &remaining[nl + 2..];
            if length == 0 {
                assert_eq!(remaining, b"\r\n");
                break;
            }
            payload.extend_from_slice(&remaining[..length]);
            assert_eq!(&remaining[length..length + 2], b"\r\n");
            remaining = &remaining[length + 2..];
        }
        let payload = String::from_utf8(payload).unwrap();
        let events: Vec<Value> = payload
            .split("\n\n")
            .filter(|s| !s.is_empty())
            .map(|block| {
                let (event, data) = block.split_once("\ndata: ").unwrap();
                let value: Value = serde_json::from_str(data).unwrap();
                assert_eq!(
                    event.strip_prefix("event: ").unwrap(),
                    value["type"].as_str().unwrap()
                );
                value
            })
            .collect();
        for (i, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], i);
        }
        events
    }

    #[test]
    fn streaming_unicode_reasoning_and_tools_match_terminal_items() {
        let r = request(json!({"input":"Hi","reasoning":{"summary":"auto"},"tools":[
            {"type":"namespace","name":"tools","tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]},
            {"type":"custom","name":"patch","description":"Apply patch"}]})).unwrap();
        let mut out = Output::new(Vec::new(), true, "gemma-4-12b", &r);
        out.start().unwrap();
        let mut native = NativeOutput::default();
        for (token, bytes) in [
            (100, b"".as_slice()),
            (5, b"thought\nThink"),
            (101, b""),
            (5, b"Answer "),
            (5, b"\xe2"),
            (5, b"\x82"),
            (5, b"\xac"),
            (48, b""),
            (5, b"call:tools.weather{city:<|\"|>Paris<|\"|>}"),
            (49, b""),
            (48, b""),
            (5, b"call:patch{input:<|\"|>a\nb<|\"|>}"),
            (49, b""),
        ] {
            native.token(token, bytes, &mut out, &r.tools).unwrap();
        }
        native.finish(&mut out, false).unwrap();
        let response = out
            .finish(12, 7, 20, native.reasoning_tokens, false)
            .unwrap();
        let events = events(&out.writer);
        assert_eq!(events.first().unwrap()["type"], "response.created");
        assert_eq!(events.last().unwrap()["type"], "response.completed");
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            7
        );
        assert_eq!(
            events.last().unwrap()["response"]["usage"],
            response["usage"]
        );
        assert_eq!(events.last().unwrap()["response"], response);
        assert!(events
            .iter()
            .filter(|e| e["type"] == "response.output_text.done")
            .all(|e| e["logprobs"] == json!([])));
        assert_eq!(response["output"][0]["summary"][0]["text"], "Think");
        assert_eq!(response["output"][1]["content"][0]["text"], "Answer €");
        assert_eq!(response["output"][2]["namespace"], "tools");
        assert_eq!(response["output"][2]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(response["output"][3]["input"], "a\nb");
        assert_eq!(response["usage"]["total_tokens"], 32);
        let done: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .map(|e| e["item"].clone())
            .collect();
        assert_eq!(json!(done), response["output"]);
        assert!(request(json!({"input":[{"role":"user","content":"Hi"},
            response["output"][0],response["output"][1],response["output"][2],response["output"][3],
            {"type":"function_call_output","call_id":response["output"][2]["call_id"],"output":"18 C"},
            {"type":"custom_tool_call_output","call_id":response["output"][3]["call_id"],"output":"done"}]})).is_ok());
    }

    #[test]
    fn token_limit_never_completes_a_partial_tool() {
        let r = request(json!({"input":"Hi"})).unwrap();
        let mut out = Output::new(Vec::new(), true, "gemma-4-12b", &r);
        out.start().unwrap();
        let mut native = NativeOutput::default();
        native.token(48, b"", &mut out, &r.tools).unwrap();
        native
            .token(5, b"call:unfinished{x:", &mut out, &r.tools)
            .unwrap();
        native.finish(&mut out, true).unwrap();
        let response = out.finish(2, 0, 3, 0, true).unwrap();
        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert_eq!(response["output"], json!([]));
        assert_eq!(
            events(&out.writer).last().unwrap()["type"],
            "response.incomplete"
        );
    }

    #[test]
    fn client_tool_search_returns_native_items_and_loads_discovered_tools() {
        let search = json!({"type":"tool_search","execution":"client","description":"Find tools",
            "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}});
        let r =
            request(json!({"input":"Find weather", "tools":[search], "parallel_tool_calls":false}))
                .unwrap();
        let mut out = Output::new(Vec::new(), true, "gemma-4-12b", &r);
        out.start().unwrap();
        out.tool("tool_search", json!({"query":"weather"}), &r.tools)
            .unwrap();
        let response = out.finish(2, 0, 4, 0, false).unwrap();
        assert_eq!(response["tools"], json!([search]));
        assert_eq!(response["parallel_tool_calls"], false);
        assert_eq!(
            response["usage"]["input_tokens_details"]["cache_write_tokens"],
            0
        );
        assert_eq!(response["output"][0]["type"], "tool_search_call");
        assert_eq!(
            response["output"][0]["arguments"],
            json!({"query":"weather"})
        );
        assert!(!events(&out.writer).iter().any(|e| e["type"]
            .as_str()
            .unwrap()
            .contains("function_call_arguments")));
        let discovered = json!({"type":"namespace","name":"weather","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]});
        let r = request(
            json!({"input":[{"role":"user","content":"Find weather"}, response["output"][0],
            {"id":"result","type":"tool_search_output","status":"completed","execution":"client",
             "call_id":response["output"][0]["call_id"],"tools":[discovered]}], "tools":[search]}),
        )
        .unwrap();
        assert_eq!(r.tools.len(), 2);
        assert_eq!(r.tools[1].prompt_name, "weather.lookup");
        assert_eq!(r.messages[1]["tool_responses"][0]["name"], "tool_search");
        assert_eq!(r.wire_tools, response_tools(json!([search, discovered])));
        assert!(request(
            json!({"input":"Hi","tools":[{"type":"tool_search","execution":"server"}]})
        )
        .is_err());
    }

    #[test]
    fn hidden_reasoning_and_generation_failure() {
        let r = request(json!({"input":"Hi"})).unwrap();
        let mut out = Output::new(Vec::new(), true, "gemma-4-12b", &r);
        out.start().unwrap();
        let mut native = NativeOutput::default();
        native.token(100, b"", &mut out, &r.tools).unwrap();
        native
            .token(5, b"thought\nsecret", &mut out, &r.tools)
            .unwrap();
        native.token(101, b"", &mut out, &r.tools).unwrap();
        native
            .token(5, b"partial answer", &mut out, &r.tools)
            .unwrap();
        out.fail("GPU failure").unwrap();
        let events = events(&out.writer);
        assert_eq!(events.last().unwrap()["type"], "response.failed");
        assert_eq!(
            events.last().unwrap()["response"]["error"]["message"],
            "GPU failure"
        );
        assert!(!String::from_utf8_lossy(&out.writer).contains("secret"));
    }

    #[test]
    fn client_disconnect_propagates_and_stops_output() {
        struct Disconnected;
        impl Write for Disconnected {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let r = request(json!({"input":"Hi"})).unwrap();
        let mut out = Output::new(Disconnected, true, "gemma-4-12b", &r);
        assert!(out.start().is_err());
    }
}
