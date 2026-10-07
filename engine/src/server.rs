//! One resident Vulkan engine, serial inference, HTTP handling provided by tiny_http.
use crate::{
    assistant::Assistant,
    generation::{self, Sampler},
    model::Engine,
    qwen,
    responses::{Defaults, Native, NativeOutput, Output, Request},
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
    /// Model name clients send (default: gemma-4-12b, or the Qwen model's name).
    #[arg(long)]
    model_id: Option<String>,
    /// Enable browser cross-origin requests.
    #[arg(long)]
    cors: bool,
    #[arg(long, default_value_t = 8192)]
    context: usize,
    #[arg(long, default_value_t = 512)]
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
    /// Drafted tokens per round (default: 3 for Gemma's assistant, 2 for Qwen's MTP layer).
    #[arg(long)]
    draft_tokens: Option<usize>,
    /// Qwen: draft only among the first N token ids (0: the whole vocabulary).
    #[arg(long, default_value_t = 65536)]
    draft_vocab: usize,
    /// Replay every prompt from position zero instead of reusing resident KV.
    #[arg(long)]
    no_prefix_cache: bool,
    /// Qwen: further machines in chain order (host:port of `trashcan qwen-node`).
    #[arg(long, value_delimiter = ',')]
    nodes: Vec<String>,
    /// Qwen: run this machine's layers on the CPU reference path.
    #[arg(long)]
    cpu: bool,
    /// Qwen: local GPU indices (default: all).
    #[arg(long, value_delimiter = ',')]
    gpus: Option<Vec<usize>>,
    /// Qwen: first layer of each stage after the first.
    #[arg(long, value_delimiter = ',')]
    cuts: Vec<usize>,
    /// Qwen: MTP layer GGUF; enables speculative decoding with --draft-tokens.
    #[arg(long)]
    mtp: Option<PathBuf>,
    /// Qwen: checkpoints kept for prefix reuse.
    #[arg(long, default_value_t = 8)]
    checkpoints: usize,
    /// Qwen: tensor parallelism, every layer split across this machine's GPUs (see `qwen-chat --tp`).
    #[arg(long)]
    tp: bool,
    /// Qwen: the other machine of the tensor-parallel group over TCP, one address per Thunderbolt link.
    #[arg(long, value_delimiter = ',')]
    tp_connect: Vec<String>,
    /// Qwen: the other machine of the tensor-parallel group over USB4STREAM devices, one per cable.
    #[arg(long, value_delimiter = ',')]
    tp_stream: Vec<String>,
}

/// A resident model behind the HTTP layer.
trait Backend {
    fn native(&self) -> Native;
    /// The template's `reasoning_effort` for a requested effort, if the model has one.
    fn effort(&self, _request: &Request) -> Option<&'static str> {
        None
    }
    fn tokenizer(&self) -> &Tokenizer;
    fn generate<W: Write>(&mut self, options: &Options, request: &Request, ids: &[u32], output: &mut Output<W>) -> Result<Value>;
    /// Forget reusable state after a failed request.
    fn reset(&mut self);
}

struct Gemma {
    engine: Engine,
    assistant: Option<Assistant>,
    tokenizer: Tokenizer,
    prefix_cache: PrefixCache,
}

impl Backend for Gemma {
    fn native(&self) -> Native {
        Native::Gemma
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn generate<W: Write>(&mut self, options: &Options, request: &Request, ids: &[u32], output: &mut Output<W>) -> Result<Value> {
        generate(
            &mut self.engine,
            &self.tokenizer,
            self.assistant.as_mut(),
            options,
            request,
            ids,
            output,
            &mut self.prefix_cache,
        )
    }
    fn reset(&mut self) {
        // Also discard a completed nonstreaming generation if its final
        // HTTP write failed, or an SSE disconnect occurred before generation.
        self.prefix_cache = PrefixCache::default();
    }
}

struct Qwen {
    chain: qwen::net::Chain,
    spec: Option<qwen::spec::Spec>,
    tokenizer: Tokenizer,
    tokens: qwen::native::Tokens,
    stop: Vec<u32>,
}

impl Backend for Qwen {
    fn native(&self) -> Native {
        Native::Qwen
    }
    fn effort(&self, request: &Request) -> Option<&'static str> {
        match request.effort.as_deref()? {
            "minimal" | "low" => Some("low"),
            "medium" => Some("medium"),
            "high" | "xhigh" | "max" => Some("xhigh"),
            _ => None,
        }
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn generate<W: Write>(&mut self, options: &Options, request: &Request, ids: &[u32], output: &mut Output<W>) -> Result<Value> {
        let p = request.sampling;
        let mut sampler = Sampler::new(p.temperature, p.top_k, p.top_p, p.seed)?;
        let e = &mut self.chain;
        if options.no_prefix_cache {
            e.forget();
        }
        let (top, cached) = e.prefill_cached(ids, &[qwen::chat::turn_start(&self.tokenizer, ids)])?;
        let mut native = qwen::native::QwenOutput::new(self.tokens, request.thinking);
        let budget = p.max_tokens.min(e.context - e.pos);
        let mut pending = sampler.sample(&top);
        let (mut generated, mut stopped) = (0, false);
        if let Some(s) = self.spec.as_mut() { s.begin_generation(); }
        'decode: while generated < budget {
            if self.stop.contains(&pending) {
                generated += 1;
                stopped = true;
                break;
            }
            let (chunk, next) = match self.spec.as_mut() {
                Some(s) if budget - generated > 1 => s.step(e, pending, &mut sampler, &self.stop, budget - generated)?,
                _ if generated + 1 < budget => {
                    let top = e.forward(&[pending], true, None)?.context("missing candidates")?;
                    (vec![pending], sampler.sample(&top))
                }
                _ => (vec![pending], pending),
            };
            for token in chunk {
                generated += 1;
                if self.stop.contains(&token) {
                    stopped = true;
                    break 'decode;
                }
                native.token(token, &self.tokenizer.decode_bytes(token, true), output, &request.tools)?;
                if !request.parallel_tool_calls && native.tool_calls > 0 {
                    stopped = true;
                    break 'decode;
                }
                if generated == budget {
                    break 'decode;
                }
            }
            pending = next;
        }
        native.finish(output, !stopped)?;
        let response = output.finish(ids.len(), cached, generated, native.reasoning_tokens, !stopped)?;
        e.checkpoint()?;
        Ok(response)
    }
    fn reset(&mut self) {
        self.chain.forget();
    }
}

/// One resident sequence, containing only tokens actually processed by the target.
/// A pending sampled token is not KV state until a forward pass consumes it.
#[derive(Default)]
struct PrefixCache {
    tokens: Vec<u32>,
    prompt_len: usize,
    logits: Vec<(f32, u32)>,
    top_k: usize,
}

impl PrefixCache {
    fn new(ids: &[u32], logits: &[(f32, u32)], top_k: usize) -> Self {
        Self {
            tokens: ids.to_vec(),
            prompt_len: ids.len(),
            logits: logits.to_vec(),
            top_k,
        }
    }
    fn plan(&self, ids: &[u32], pos: usize, top_k: usize, draft: bool) -> (usize, bool) {
        if self.tokens.len() != pos || ids.is_empty() {
            return (0, false);
        }
        let common = self
            .tokens
            .iter()
            .zip(ids)
            .take_while(|(a, b)| a == b)
            .count();
        // The original prompt logits remain exact across repeated requests. A
        // drafter also needs the final hidden row, so it must replay a token.
        let saved_logits = !draft
            && common == ids.len()
            && ids.len() == self.prompt_len
            && self.top_k == top_k
            && !self.logits.is_empty();
        (
            if saved_logits {
                common
            } else {
                common.min(ids.len() - 1)
            },
            saved_logits,
        )
    }
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

pub fn serve(mut options: Options) -> Result<()> {
    ensure!(options.max_tokens > 0, "max-tokens must be positive");
    Sampler::new(
        options.temperature,
        options.top_k,
        options.top_p,
        options.seed,
    )?;
    let gguf = crate::gguf::Gguf::open(&options.model)?;
    if gguf.get_str("general.architecture")? == "qwen35" {
        let name = gguf.get_str("general.name").unwrap_or("qwen").to_lowercase().replace(' ', "-");
        options.model_id.get_or_insert(name);
        drop(gguf);
        let devices = qwen::chat::devices(options.cpu, options.gpus.clone())?;
        let draft = options.mtp.is_some();
        let k = options.draft_tokens.unwrap_or(2);
        let verify = if draft { k + 1 } else { 0 };
        let mut chain = if options.tp || !options.tp_connect.is_empty() || !options.tp_stream.is_empty() {
            anyhow::ensure!(options.nodes.is_empty() && !options.cpu && options.cuts.is_empty(), "tensor parallelism runs without --nodes, --cpu or --cuts");
            let qwen::pipeline::Devices::Gpus(gpus) = devices else { unreachable!() };
            let peer = qwen::tp::peer(&options.tp_connect, &options.tp_stream)?;
            let mtp = options.mtp.as_deref();
            qwen::net::Chain::tensor_parallel(&options.model, &gpus, peer, options.context, options.batch, verify, mtp, options.draft_vocab)?
        } else {
            qwen::net::Chain::new(
                &options.model,
                devices,
                &options.nodes,
                &options.cuts,
                options.context,
                options.batch,
                verify,
                options.mtp.as_deref(),
                draft,
            )?
        };
        chain.max_checkpoints = options.checkpoints;
        let mut spec = if draft { Some(qwen::spec::Spec::new(&chain, k)?) } else { None };
        if let Some(s) = spec.as_mut() {
            s.vocab = options.draft_vocab;
        }
        let tokenizer = Tokenizer::new(&chain.local.gguf)?;
        let backend = Qwen {
            tokens: qwen::native::Tokens::new(&tokenizer)?,
            stop: qwen::chat::Specials::new(&tokenizer).stop,
            chain,
            spec,
            tokenizer,
        };
        return run(options, backend);
    }
    drop(gguf);
    options.draft_tokens.get_or_insert(3);
    crate::validate_draft(options.draft.as_ref(), options.draft_tokens.unwrap(), options.batch)?;
    options.model_id.get_or_insert("gemma-4-12b".into());
    let mut engine = Engine::new(
        &options.model,
        options.context,
        options.batch,
        options.cpu_reduce,
    )?;
    // Buffers hold 64 candidates; greedy requests can cache a separate one-candidate graph.
    engine.set_top_k(64)?;
    let tokenizer = Tokenizer::new(&engine.gguf)?;
    let assistant = options
        .draft
        .as_ref()
        .map(|path| Assistant::new(path, &engine, 64))
        .transpose()?;
    run(options, Gemma { engine, assistant, tokenizer, prefix_cache: PrefixCache::default() })
}

fn run(options: Options, mut backend: impl Backend) -> Result<()> {
    let model_id = options.model_id.clone().unwrap_or_default();
    ensure!(!model_id.is_empty(), "model-id must be nonempty");
    let http = tiny_http::Server::http((options.host.as_str(), options.port))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let defaults = Defaults {
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        seed: options.seed,
        thinking: options.thinking,
    };
    eprintln!(
        "Responses server listening on http://{} (model: {model_id})",
        http.server_addr(),
    );
    for request in http.incoming_requests() {
        if let Err(e) = handle(request, &options, &model_id, defaults, &mut backend) {
            backend.reset();
            eprintln!("request failed: {e:#}");
        }
    }
    Ok(())
}

fn prepare(
    request: &mut tiny_http::Request,
    options: &Options,
    model_id: &str,
    defaults: Defaults,
    backend: &impl Backend,
) -> Result<(Request, Vec<u32>)> {
    let (native, tokenizer, effort) = (backend.native(), backend.tokenizer(), |r: &Request| backend.effort(r));
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
    let parsed = Request::parse_native(&value, defaults, model_id, native)?;
    let tools: Vec<_> = parsed.tools.iter().map(|t| t.schema.clone()).collect();
    let mut rendered = tokenizer.render_with_effort(&parsed.messages, parsed.thinking, &tools, effort(&parsed))?;
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
    model_id: &str,
    defaults: Defaults,
    backend: &mut impl Backend,
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
                json!({"object":"list","data":[{"id":model_id,"object":"model","created":0,"owned_by":"trashcan"}]}),
                options.cors,
            )
        }
        (&Method::Post, "/v1/responses" | "/responses") => {}
        _ => return error(request, 404, "unknown endpoint", options.cors),
    }
    let (parsed, ids) = match prepare(&mut request, options, model_id, defaults, backend) {
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
        let mut output = Output::new(writer, true, model_id, &parsed);
        output.start()?;
        if let Err(e) = backend.generate(options, &parsed, &ids, &mut output) {
            // If the client disconnected this write fails too; inference has already stopped.
            let _ = output.fail(&format!("{e:#}"));
            return Err(e);
        }
    } else {
        let mut output = Output::new(std::io::sink(), false, model_id, &parsed);
        match backend.generate(options, &parsed, &ids, &mut output) {
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
    prefix_cache: &mut PrefixCache,
) -> Result<Value> {
    // Keep the shared cache empty until the complete request succeeds. Any GPU,
    // parsing or socket failure then forces the next request to rebuild its KV.
    let previous = std::mem::take(prefix_cache);
    let params = request.sampling;
    let top_k = if params.temperature == 0. { 1 } else { 64 };
    if let Some(draft) = assistant.as_mut() {
        draft.reset();
        draft.set_top_k(top_k)?;
    }
    engine.set_top_k(top_k)?;
    let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p, params.seed)?;
    let (candidate, saved_logits) = previous.plan(ids, engine.pos, top_k, assistant.is_some());
    let cached =
        if !options.no_prefix_cache && candidate > 0 && engine.rewind_prefix(candidate).is_ok() {
            candidate
        } else {
            engine.pos = 0;
            0
        };
    let logits = if cached > 0 && saved_logits {
        previous.logits
    } else {
        generation::prefill(engine, &ids[cached..], options.batch)?
    };
    let mut next_cache = PrefixCache::new(ids, &logits, top_k);
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
            .unwrap_or(3)
            .min(budget - generated - 1)
            .min(engine.remaining().saturating_sub(1));
        if let Some(draft) = assistant.as_mut().filter(|_| k > 0) {
            let (chunk, next) = draft.step(engine, pending, &mut sampler, k)?;
            // step returns only the committed prefix, after rejected proposals
            // have been rewound. Even tokens after an early protocol stop were
            // actually processed and therefore belong to the resident sequence.
            next_cache.tokens.extend_from_slice(&chunk);
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
                let logits = engine.forward(&[pending], None, false)?.remove(0);
                next_cache.tokens.push(pending);
                pending = sampler.sample(&logits);
            }
        }
    }
    native.finish(output, !stopped)?;
    let response = output.finish(
        ids.len(),
        cached,
        generated,
        native.reasoning_tokens,
        !stopped,
    )?;
    if !options.no_prefix_cache && next_cache.tokens.len() == engine.pos {
        *prefix_cache = next_cache;
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;

    #[test]
    fn repeated_prompt_reuses_saved_logits_and_excludes_unprocessed_output() {
        let mut cache = PrefixCache::new(&[2, 3, 4], &[(1., 5)], 1);
        // Token 5 was forwarded; a final sampled token 6 was only emitted.
        cache.tokens.push(5);
        assert_eq!(cache.plan(&[2, 3, 4], 4, 1, false), (3, true));
        assert_eq!(cache.plan(&[2, 3, 4, 5, 6, 7], 4, 1, false), (4, false));
        assert_eq!(cache.plan(&[2, 3, 4], 5, 1, false), (0, false));
    }

    #[test]
    fn changed_sampler_or_drafter_recomputes_the_last_prompt_token() {
        let cache = PrefixCache::new(&[2, 3, 4], &[(1., 5)], 1);
        assert_eq!(cache.plan(&[2, 3, 4], 3, 64, false), (2, false));
        assert_eq!(cache.plan(&[2, 3, 4], 3, 1, true), (2, false));
        let sampled = PrefixCache::new(&[2, 3, 4], &[(1., 5), (0., 6)], 64);
        assert_eq!(sampled.plan(&[2, 3, 4], 3, 64, false), (3, true));
    }

    #[test]
    fn exact_prefix_matching_handles_changes_shortening_and_empty_prompts() {
        let cache = PrefixCache::new(&[2, 3, 4, 5], &[(1., 6)], 1);
        assert_eq!(cache.plan(&[2, 3], 4, 1, false), (1, false));
        assert_eq!(cache.plan(&[2, 3, 9, 5], 4, 1, false), (2, false));
        assert_eq!(cache.plan(&[9, 3, 4, 5], 4, 1, false), (0, false));
        assert_eq!(cache.plan(&[], 4, 1, false), (0, false));
        assert_eq!(PrefixCache::default().plan(&[2], 0, 1, false), (0, false));
    }

    #[test]
    fn committed_speculative_tokens_include_stop_and_unemitted_tail() {
        let mut cache = PrefixCache::new(&[2, 3, 4], &[(1., 5)], 1);
        // All returned tokens were processed, even if a tool-call stop caused
        // protocol output to stop after token 5. Rejected proposals are absent.
        cache.tokens.extend_from_slice(&[5, 6, 106]);
        assert_eq!(cache.plan(&[2, 3, 4, 5, 7], 6, 1, true), (4, false));
        assert_eq!(cache.plan(&[2, 3, 4, 5, 6, 106, 8], 6, 1, true), (6, false));
    }

    #[test]
    fn taking_cache_invalidates_failures_until_successful_replacement() {
        let mut cache = PrefixCache::new(&[2, 3], &[(1., 4)], 1);
        let previous = std::mem::take(&mut cache);
        assert_eq!(previous.plan(&[2, 3], 2, 1, false), (2, true));
        assert_eq!(cache.plan(&[2, 3], 2, 1, false), (0, false));
    }
}
