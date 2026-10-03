//! One resident Vulkan engine, serial inference, HTTP handling provided by tiny_http.
use crate::{
    assistant::Assistant,
    generation::{self, Sampler},
    model::Engine,
    responses::{Defaults, NativeOutput, Output, Request},
    tokenizer::Tokenizer,
};
use anyhow::{ensure, Context, Result};
use clap::Args;
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    path::PathBuf,
};
use tiny_http::{Header, Method, Response, StatusCode};

const MAX_BODY: u64 = 8 * 1024 * 1024;

#[derive(Args)]
pub struct Options {
    pub model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value = "gemma-4-12b")]
    model_id: String,
    /// Enable browser cross-origin requests.
    #[arg(long)]
    cors: bool,
    #[arg(long, default_value_t = 8192)]
    context: usize,
    #[arg(long, default_value_t = 32)]
    batch: usize,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long)]
    thinking: bool,
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 64)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long)]
    cpu_reduce: bool,
    #[arg(long)]
    draft: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    draft_tokens: usize,
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name, value).unwrap()
}

fn respond(request: tiny_http::Request, status: u16, value: Value, cors: bool) -> Result<()> {
    let mut response = Response::from_string(value.to_string())
        .with_status_code(StatusCode(status))
        .with_header(header("Content-Type", "application/json"));
    if cors {
        response.add_header(header("Access-Control-Allow-Origin", "*"));
    }
    request.respond(response)?;
    Ok(())
}

fn error(request: tiny_http::Request, status: u16, message: &str, cors: bool) -> Result<()> {
    respond(
        request,
        status,
        json!({"error":{"message":message,"type":if status < 500 {"invalid_request_error"} else {"server_error"},"param":null,"code":null}}),
        cors,
    )
}

pub fn serve(options: Options) -> Result<()> {
    crate::validate_draft(options.draft.as_ref(), options.draft_tokens, options.batch)?;
    ensure!(options.max_tokens > 0, "max-tokens must be positive");
    ensure!(!options.model_id.is_empty(), "model-id must be nonempty");
    Sampler::new(
        options.temperature,
        options.top_k,
        options.top_p,
        options.seed,
    )?;
    let http = tiny_http::Server::http((options.host.as_str(), options.port))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut engine = Engine::new(
        &options.model,
        options.context,
        options.batch,
        options.cpu_reduce,
    )?;
    // Candidate capacity is fixed once graphs are recorded. Requests can sample any top-k up to 64.
    engine.set_top_k(64)?;
    let tokenizer = Tokenizer::new(&engine.gguf)?;
    let mut assistant = options
        .draft
        .as_ref()
        .map(|path| Assistant::new(path, &engine, 64))
        .transpose()?;
    let defaults = Defaults {
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        seed: options.seed,
        thinking: options.thinking,
    };
    eprintln!(
        "Responses server listening on http://{} (model: {})",
        http.server_addr(),
        options.model_id
    );
    for request in http.incoming_requests() {
        if let Err(e) = handle(
            request,
            &options,
            defaults,
            &mut engine,
            &tokenizer,
            assistant.as_mut(),
        ) {
            // Failed writes cancel generation, and the next request starts from a clean KV position.
            eprintln!("request failed: {e:#}");
        }
    }
    Ok(())
}

fn prepare(
    request: &mut tiny_http::Request,
    options: &Options,
    defaults: Defaults,
    tokenizer: &Tokenizer,
) -> Result<(Request, Vec<u32>)> {
    ensure!(
        request.body_length().is_none_or(|n| n as u64 <= MAX_BODY),
        "request body exceeds 8 MiB"
    );
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut body)?;
    ensure!(body.len() as u64 <= MAX_BODY, "request body exceeds 8 MiB");
    let value = serde_json::from_slice(&body).context("invalid JSON request")?;
    let parsed = Request::parse(&value, defaults, &options.model_id)?;
    let tools: Vec<_> = parsed.tools.iter().map(|t| t.schema.clone()).collect();
    let mut rendered = tokenizer.render_with_tools(&parsed.messages, parsed.thinking, &tools)?;
    // Tool results continue the same model turn. Some embedded templates terminate it anyway.
    if parsed
        .messages
        .last()
        .is_some_and(|m| m.get("tool_responses").is_some())
        && rendered.ends_with("<turn|>\n")
    {
        rendered.push_str("<|turn>model\n");
    }
    let ids = tokenizer.encode(&rendered, true, true);
    ensure!(
        ids.len() < options.context,
        "prompt has {} tokens; context is {} and needs room for output",
        ids.len(),
        options.context
    );
    Ok((parsed, ids))
}

fn handle(
    mut request: tiny_http::Request,
    options: &Options,
    defaults: Defaults,
    engine: &mut Engine,
    tokenizer: &Tokenizer,
    assistant: Option<&mut Assistant>,
) -> Result<()> {
    let path = request.url().split('?').next().unwrap_or("").to_owned();
    if request.method() == &Method::Options && options.cors {
        let allowed_headers = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Access-Control-Request-Headers"))
            .map(|h| h.value.as_str())
            .unwrap_or("Content-Type, Authorization");
        let response = Response::empty(StatusCode(204))
            .with_header(header("Access-Control-Allow-Origin", "*"))
            .with_header(header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"))
            .with_header(header("Access-Control-Allow-Headers", allowed_headers));
        request.respond(response)?;
        return Ok(());
    }
    match (request.method(), path.as_str()) {
        (&Method::Get, "/health") => {
            return respond(request, 200, json!({"status":"ok"}), options.cors)
        }
        (&Method::Get, "/v1/models" | "/models") => {
            return respond(
                request,
                200,
                json!({"object":"list","data":[{"id":options.model_id,"object":"model","created":0,"owned_by":"trashcan"}]}),
                options.cors,
            )
        }
        (&Method::Post, "/v1/responses" | "/responses") => {}
        _ => return error(request, 404, "unknown endpoint", options.cors),
    }
    let (parsed, ids) = match prepare(&mut request, options, defaults, tokenizer) {
        Ok(value) => value,
        Err(e) => return error(request, 400, &format!("{e:#}"), options.cors),
    };
    if parsed.stream {
        let mut writer = request.into_writer();
        write!(writer, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n")?;
        if options.cors {
            write!(writer, "Access-Control-Allow-Origin: *\r\n")?;
        }
        write!(writer, "\r\n")?;
        writer.flush()?;
        let mut output = Output::new(writer, true, &options.model_id, &parsed);
        output.start()?;
        if let Err(e) = generate(
            engine,
            tokenizer,
            assistant,
            options,
            &parsed,
            &ids,
            &mut output,
        ) {
            // If the client disconnected this write fails too; inference has already stopped.
            let _ = output.fail(&format!("{e:#}"));
            return Err(e);
        }
    } else {
        let mut output = Output::new(std::io::sink(), false, &options.model_id, &parsed);
        match generate(
            engine,
            tokenizer,
            assistant,
            options,
            &parsed,
            &ids,
            &mut output,
        ) {
            Ok(response) => respond(request, 200, response, options.cors)?,
            Err(e) => error(request, 500, &format!("{e:#}"), options.cors)?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Resident engine and request-specific protocol state.
fn generate<W: Write>(
    engine: &mut Engine,
    tokenizer: &Tokenizer,
    mut assistant: Option<&mut Assistant>,
    options: &Options,
    request: &Request,
    ids: &[u32],
    output: &mut Output<W>,
) -> Result<Value> {
    engine.pos = 0;
    if let Some(draft) = assistant.as_mut() {
        draft.reset();
    }
    let params = request.sampling;
    let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p, params.seed)?;
    let logits = generation::prefill(engine, ids, options.batch)?;
    let mut pending = sampler.sample(&logits);
    let budget = params.max_tokens.min(engine.remaining());
    let mut native = NativeOutput::default();
    let mut generated = 0;
    let mut stopped = false;
    'decode: while generated < budget {
        if matches!(pending, 1 | 106 | 50) {
            generated += 1;
            stopped = true;
            break;
        }
        let k = options
            .draft_tokens
            .min(budget - generated - 1)
            .min(engine.remaining().saturating_sub(1));
        if let Some(draft) = assistant.as_mut().filter(|_| k > 0) {
            let (chunk, next) = draft.step(engine, pending, &mut sampler, k)?;
            pending = next;
            for token in chunk {
                generated += 1;
                if matches!(token, 1 | 106 | 50) {
                    stopped = true;
                    break 'decode;
                }
                native.token(
                    token,
                    &tokenizer.decode_bytes(token, true),
                    output,
                    &request.tools,
                )?;
                if !request.parallel_tool_calls && native.tool_calls > 0 {
                    stopped = true;
                    break 'decode;
                }
            }
        } else {
            generated += 1;
            native.token(
                pending,
                &tokenizer.decode_bytes(pending, true),
                output,
                &request.tools,
            )?;
            if !request.parallel_tool_calls && native.tool_calls > 0 {
                stopped = true;
                break;
            }
            if generated < budget {
                pending = sampler.sample(&engine.forward(&[pending], None, false)?.remove(0));
            }
        }
    }
    native.finish(output, !stopped)?;
    output.finish(ids.len(), generated, native.reasoning_tokens, !stopped)
}
