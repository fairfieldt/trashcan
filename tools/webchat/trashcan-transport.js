// The pipe web UI (piPE web/, opened with ?transport=/trashcan-transport.js) on a trashcan server, without a pi
// process: this module answers the agent methods of piPE docs/rpc.md in the page and runs the agent loop itself.
// Each round is one POST /v1/responses (passed through to `trashcan serve` by server.py); the tools the model calls
// run through server.py's POST /tool. Responses stream events become pi's stream events (thinking/text/toolcall
// start, delta, end), and the transcript is kept as pi's unified messages, one session per conversation in
// localStorage. model.set switches what the cluster serves (server.py POST /switch runs cluster.sh): Flash-Next on six
// Macs or Qwen3.8-27B on four (30 s to 3 min measured).

const INSTRUCTIONS = `You are a helpful assistant running on a cluster of six 2013 Mac Pros (trashcan).
Tools: web_search, fetch_page, and bash (a shell on the cluster's first Mac).

## Answering questions
When asked a factual question, run 1-3 web_search calls first (fetch_page when the snippets are not enough), then
answer. Structure the answer: the direct answer first, then the key details in short sections or bullets, and the
sources you used at the end.`;

const TOOLS = [
  {
    type: 'function', name: 'web_search',
    description: 'Search the web. Returns titles, URLs and snippets.',
    parameters: { type: 'object', properties: { query: { type: 'string' }, num_results: { type: 'integer' } }, required: ['query'] },
  },
  {
    type: 'function', name: 'fetch_page',
    description: 'Read a web page as plain text (the first 8000 characters).',
    parameters: { type: 'object', properties: { url: { type: 'string' } }, required: ['url'] },
  },
  {
    type: 'function', name: 'bash',
    description: 'Run a shell command on the cluster\'s first Mac (Debian, 60 s limit). Returns stdout and stderr.',
    parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] },
  },
];
const LEVELS = ['off', 'low', 'medium', 'high'];
const MAX_ROUNDS = 10;
const STORE = 'trashcan.sessions';
// What the cluster can serve: the model id its server reports, the recipe that starts it, the server's context.
const MODELS = [
  { id: 'qwen3.8-flash-next', name: 'Qwen3.8-Flash-Next · 6 Macs', recipe: '6-flashnext', context: 32768 },
  { id: 'qwen3.8-27b', name: 'Qwen3.8-27B · 4 Macs', recipe: '4-qwen27b', context: 8192 },
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const now = () => Date.now();
const uid = () => now().toString(36) + Math.random().toString(36).slice(2, 8);
const rpcError = (code, message) => Object.assign(new Error(message), { code });

function loadSessions() {
  try { return JSON.parse(localStorage.getItem(STORE) || '[]'); } catch (e) { return []; }
}

class TrashcanTransport {
  constructor() {
    this.handlers = [];
    this.sessions = loadSessions();          // [{id, path, createdAt, mtime, title, messages}]
    this.current = null;
    this.running = false;
    this.controller = null;
    this.thinking = localStorage.getItem('trashcan.thinking') || 'medium';
    this.modelId = null;
    this.switching = null;
    this.contextTokens = 0;
  }

  onEvent(fn) { this.handlers.push(fn); }
  emit(name, data) { for (const fn of this.handlers) { try { fn(name, data); } catch (e) { console.error(e); } } }

  async ready() {
    this.current = this.sessions[0] || this.newSession();
    // Opened mid-switch: show the model being started and follow the switch in the background.
    const st = await fetch('/switch').then((x) => x.json()).catch(() => ({}));
    const target = st.state === 'switching' && MODELS.find((m) => m.recipe === st.recipe);
    if (target) {
      this.modelId = target.id;
      this.switching = target.name;
      this.followSwitch(target).catch((e) => this.emit('notify', { message: e.message, level: 'error' }));
    } else {
      const r = await fetch('/v1/models');
      if (!r.ok) throw new Error('no trashcan server behind this page');
      const j = await r.json();
      this.modelId = j.data && j.data[0] && j.data[0].id;
      if (!this.modelId) throw new Error('the trashcan server lists no model');
    }
    queueMicrotask(() => this.emit('state', this.state()));
  }

  close() { if (this.controller) this.controller.abort(); }
  notify() {}

  async send(req) {
    try {
      return { id: req.id, result: await this.handle(req.method, req.params || {}) };
    } catch (e) {
      return { id: req.id, error: { code: e.code || 500, message: e.message } };
    }
  }

  // ------------------------------------------------------------------ state
  model(id = this.modelId) {
    const known = MODELS.find((m) => m.id === id);
    const ctx = known ? known.context : 8192;
    return {
      provider: 'trashcan', id, name: known ? known.name : id, api: 'openai-responses', contextWindow: ctx,
      maxTokens: Math.min(8192, Math.floor(ctx / 4)), reasoning: true, thinkingLevels: LEVELS, supportsFast: false,
      configured: true, visible: true,
    };
  }
  sessionRef(s) { return { path: s.path, id: s.id, cwd: 'trashcan' }; }
  state() {
    return {
      running: this.running, model: this.model(), thinking: this.thinking, thinkingEffective: this.thinking, fast: false,
      session: this.sessionRef(this.current), contextTokens: this.contextTokens, cost: 0,
      messages: this.current.messages.length, cwd: 'trashcan', home: '', mode: 'native',
    };
  }
  save() {
    this.current.mtime = now();
    const first = this.current.messages.find((m) => m.role === 'user');
    if (first && !this.current.title) this.current.title = textOf(first).slice(0, 80);
    this.sessions = [this.current, ...this.sessions.filter((s) => s.id !== this.current.id)].filter((s) => s.messages.length || s === this.current);
    try { localStorage.setItem(STORE, JSON.stringify(this.sessions.slice(0, 50))); } catch (e) { /* full: keep in memory */ }
  }
  newSession() {
    const id = uid();
    const s = { id, path: `trashcan/${id}`, createdAt: new Date().toISOString(), mtime: now(), title: '', messages: [] };
    this.sessions.unshift(s);
    this.contextTokens = 0;
    return s;
  }
  find(path) {
    const s = this.sessions.find((x) => x.path === path);
    if (!s) throw rpcError(404, 'no such session');
    return s;
  }
  idle() {
    if (this.running) throw rpcError(409, 'a turn is running');
    if (this.switching) throw rpcError(409, `switching to ${this.switching}`);
  }

  // Ask server.py to start the other model and wait until the trashcan server answers with it.
  async switchTo(id) {
    const target = MODELS.find((m) => m.id === id);
    if (!target) throw rpcError(404, `unknown model ${id}`);
    if (id === this.modelId) return;
    this.idle();
    const r = await fetch('/switch', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ recipe: target.recipe }) });
    if (!r.ok) throw rpcError(r.status, (await r.json()).error || `switch: HTTP ${r.status}`);
    const before = this.modelId;
    this.modelId = id;             // the picker and header show the choice at once; reverted if it fails
    this.switching = target.name;
    this.emit('state', this.state());
    this.emit('notify', { message: `Starting ${target.name}. This can take a few minutes.`, level: 'info' });
    this.followSwitch(target).catch((e) => {
      this.modelId = before;
      this.emit('state', this.state());
      this.emit('notify', { message: e.message, level: 'error' });
    });
  }

  // Poll server.py's /switch until the target serves, telling the user what it waits for.
  async followSwitch(target) {
    const PHASES = [
      [/^waiting for (.+)$/, (m) => `Waiting for ${m[1]}`],
      [/^preflight .*: OK/, () => 'Checking the Macs: OK'],
      [/^start /, () => 'Loading the model'],
    ];
    let said = '';
    try {
      for (;;) {
        await sleep(2000);
        const st = await (await fetch('/switch')).json();
        const lines = (st.log || '').trim().split('\n');
        if (st.state === 'error') throw new Error(`Could not start ${target.name}: ${lines.slice(-2).join(' ')}`);
        if (st.state === 'idle') break;
        for (let i = lines.length - 1; i >= 0; i--) {
          const hit = PHASES.find(([re]) => re.test(lines[i]));
          if (!hit) continue;
          const msg = hit[1](lines[i].match(hit[0]));
          if (msg !== said) { said = msg; this.emit('notify', { message: msg, level: 'info' }); }
          break;
        }
      }
      const j = await (await fetch('/v1/models')).json();
      const id = j.data && j.data[0] && j.data[0].id;
      if (id !== target.id) throw new Error(`The server came up with ${id}, not ${target.id}`);
      this.modelId = id;
      this.contextTokens = 0;
      this.emit('notify', { message: `${target.name} is ready`, level: 'info' });
    } finally {
      this.switching = null;
      this.emit('state', this.state());
    }
  }

  async handle(method, p) {
    switch (method) {
      case 'hello':
        return { protocol: 1, name: 'trashcan', version: 'trashcan-transport', state: this.state(), commands: [], tools: TOOLS.map((t) => t.name), extensions: 0, local: { available: false }, app: null, capsules: [] };
      case 'state': return this.state();
      case 'messages.list': return { messages: this.current.messages, running: this.running, queued: [] };
      case 'model.list': return { models: MODELS.map((m) => this.model(m.id)), current: `trashcan/${this.modelId}`, selection: null };
      case 'model.set':
        await this.switchTo(String(p.model || '').replace(/^trashcan\//, ''));
        return { model: `trashcan/${this.modelId}` };   // the switch continues in the background
      case 'model.select': return { models: p.models || null };
      case 'thinking.set':
        this.idle();
        this.thinking = LEVELS.includes(p.level) ? p.level : 'medium';
        localStorage.setItem('trashcan.thinking', this.thinking);
        this.emit('state', this.state());
        return { thinking: this.thinking };
      case 'fast.set': return { fast: false, supported: false };
      case 'commands.list': return { commands: [] };
      case 'session.info': return this.sessionRef(this.current);
      case 'session.list':
        return {
          sessions: this.sessions.filter((s) => s.messages.length).map((s) => ({
            path: s.path, name: s.title || 'New conversation', id: s.id, createdAt: s.createdAt, cwd: 'trashcan',
            mtime: s.mtime, size: JSON.stringify(s.messages).length, title: s.title, messages: s.messages.length,
            lastRole: s.messages.length ? s.messages[s.messages.length - 1].role : undefined,
          })),
          current: this.current.path,
        };
      case 'session.read': return { messages: this.find(p.path).messages };
      case 'session.new':
        this.idle();
        this.current = this.newSession();
        this.emit('state', this.state());
        return { session: this.state(), messages: [] };
      case 'session.open':
        this.idle();
        this.current = this.find(p.path);
        this.contextTokens = 0;
        this.emit('state', this.state());
        return { session: this.state(), messages: this.current.messages };
      case 'session.delete': {
        this.idle();
        const gone = this.find(p.path);
        this.sessions = this.sessions.filter((s) => s !== gone);
        let switched = false;
        if (gone === this.current) {
          this.current = p.switchTo ? this.find(p.switchTo) : this.newSession();
          switched = true;
        }
        try { localStorage.setItem(STORE, JSON.stringify(this.sessions)); } catch (e) { /* ignore */ }
        this.emit('state', this.state());
        return switched ? { deleted: p.path, switched, embedded: false, session: this.state(), messages: this.current.messages } : { deleted: p.path, switched, embedded: false };
      }
      case 'agent.prompt': {
        const text = String(p.text || '').trim();
        if (!text) throw rpcError(400, 'empty prompt');
        this.idle();
        this.turn(text);
        return { kind: 'prompt', queued: false };
      }
      case 'agent.abort': {
        const was = this.running;
        if (this.controller) this.controller.abort();
        return { aborted: was };
      }
      case 'image.info': return { carries: {}, live: { model: `trashcan/${this.modelId}` } };
      case 'keys.list': return { providers: [] };
      case 'mcp.list': return { servers: [], scopes: { user: '', project: '' } };
      case 'auth.status': return { provider: 'openai-codex', loggedIn: false, pending: false, canLogin: false };
      case 'apps.list': return { apps: [] };
      case 'app.info': return {};
      default: throw rpcError(-32601, `not available with trashcan: ${method}`);
    }
  }

  // ------------------------------------------------------------- the agent
  async turn(text) {
    this.running = true;
    this.controller = new AbortController();
    const user = { role: 'user', content: [{ type: 'text', text }], timestamp: now() };
    this.current.messages.push(user);
    this.save();
    this.emit('turn.start', { text });
    this.emit('message', { message: user });
    this.emit('state', this.state());
    let error = null;
    try {
      for (let n = 0; n < MAX_ROUNDS; n++) {
        const msg = await this.round();
        this.current.messages.push(msg);
        this.save();
        this.emit('message', { message: msg });
        if (msg.stopReason !== 'toolUse') break;
        for (const call of msg.content.filter((b) => b.type === 'toolCall')) {
          this.emit('tool.start', { name: call.name, args: call.arguments });
          const { text: out, isError } = await this.runTool(call);
          const content = [{ type: 'text', text: out }];
          this.emit('tool.end', { name: call.name, content, details: null, isError });
          const result = { role: 'toolResult', toolCallId: call.id, toolName: call.name, content, isError, timestamp: now() };
          this.current.messages.push(result);
          this.save();
          this.emit('message', { message: result });
        }
      }
    } catch (e) {
      error = e.name === 'AbortError' ? 'aborted' : e.message;
      if (e.name !== 'AbortError') this.emit('stream', { type: 'error', index: 0, message: e.message });
    } finally {
      this.running = false;
      this.controller = null;
      this.emit('turn.done', { error });
      this.emit('state', this.state());
      this.emit('settled', {});
    }
  }

  // pi's unified messages -> Responses input items.
  input() {
    const items = [];
    for (const m of this.current.messages) {
      if (m.role === 'user') {
        items.push({ role: 'user', content: [{ type: 'input_text', text: textOf(m) }] });
      } else if (m.role === 'assistant') {
        const text = m.content.filter((b) => b.type === 'text').map((b) => b.text).join('');
        if (text) items.push({ role: 'assistant', content: [{ type: 'output_text', text }] });
        for (const b of m.content) {
          if (b.type === 'toolCall') items.push({ type: 'function_call', call_id: b.id, name: b.name, arguments: JSON.stringify(b.arguments || {}) });
        }
      } else if (m.role === 'toolResult') {
        items.push({ type: 'function_call_output', call_id: m.toolCallId, output: textOf(m) });
      }
    }
    // A tool call left without its result (a turn stopped mid-tool) cannot be sent back.
    const answered = new Set(items.filter((i) => i.type === 'function_call_output').map((i) => i.call_id));
    return items.filter((i) => i.type !== 'function_call' || answered.has(i.call_id));
  }

  // One server round, streamed: returns the finished assistant message.
  async round() {
    const body = {
      model: this.modelId, instructions: INSTRUCTIONS, input: this.input(), tools: TOOLS, stream: true, store: false,
      reasoning: this.thinking === 'off' ? { effort: 'none' } : { effort: this.thinking, summary: 'auto' },
      max_output_tokens: this.model().maxTokens,
    };
    const r = await fetch('/v1/responses', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: this.controller.signal,
    });
    if (!r.ok || !r.body) {
      let msg = `trashcan answered HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not JSON */ }
      throw new Error(msg);
    }
    const msg = {
      role: 'assistant', content: [], api: 'openai-responses', provider: 'trashcan', model: this.modelId,
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      stopReason: 'stop', timestamp: now(),
    };
    const blocks = new Map();   // Responses item id (+ part) -> {index, block, kind}
    const open = (key, kind, block) => {
      const index = msg.content.length;
      msg.content.push(block);
      const b = { index, block, kind };
      blocks.set(key, b);
      this.emit('stream', { type: `${kind}_start`, index, block: { ...block } });
      return b;
    };
    const close = (b) => { if (b && !b.closed) { b.closed = true; this.emit('stream', { type: `${b.kind}_end`, index: b.index, block: { ...b.block } }); } };
    let final = null;
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, k);
        buf = buf.slice(k + 2);
        const data = chunk.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).join('');
        if (!data || data === '[DONE]') continue;
        let ev;
        try { ev = JSON.parse(data); } catch (e) { continue; }
        const t = ev.type;
        if (t === 'response.reasoning_summary_text.delta') {
          const key = `${ev.item_id}:r`;
          const b = blocks.get(key) || open(key, 'thinking', { type: 'thinking', thinking: '' });
          b.block.thinking += ev.delta || '';
          this.emit('stream', { type: 'thinking_delta', index: b.index, delta: ev.delta || '' });
        } else if (t === 'response.output_item.done' && ev.item && ev.item.type === 'reasoning') {
          close(blocks.get(`${ev.item.id}:r`));
        } else if (t === 'response.output_text.delta') {
          const key = `${ev.item_id}:t`;
          const b = blocks.get(key) || open(key, 'text', { type: 'text', text: '' });
          b.block.text += ev.delta || '';
          this.emit('stream', { type: 'text_delta', index: b.index, delta: ev.delta || '' });
        } else if (t === 'response.output_text.done') {
          close(blocks.get(`${ev.item_id}:t`));
        } else if (t === 'response.output_item.added' && ev.item && ev.item.type === 'function_call') {
          open(`${ev.item.id}:f`, 'toolcall', { type: 'toolCall', id: ev.item.call_id, name: ev.item.name, arguments: {} });
        } else if (t === 'response.function_call_arguments.delta') {
          const b = blocks.get(`${ev.item_id}:f`);
          if (b) {
            b.raw = (b.raw || '') + (ev.delta || '');
            b.block.arguments = partialJson(b.raw);
            this.emit('stream', { type: 'toolcall_delta', index: b.index, delta: ev.delta || '', block: { ...b.block } });
          }
        } else if (t === 'response.output_item.done' && ev.item && ev.item.type === 'function_call') {
          const b = blocks.get(`${ev.item.id}:f`);
          if (b) {
            try { b.block.arguments = JSON.parse(ev.item.arguments || '{}'); } catch (e) { b.block.arguments = {}; }
            b.block.id = ev.item.call_id;
            close(b);
          }
        } else if (t === 'response.completed' || t === 'response.incomplete') {
          final = ev.response;
        } else if (t === 'response.failed') {
          throw new Error((ev.response && ev.response.error && ev.response.error.message) || 'generation failed');
        }
      }
    }
    if (!final) throw new Error('the response ended early');
    for (const b of blocks.values()) close(b);
    const u = final.usage || {};
    msg.usage.input = u.input_tokens || 0;
    msg.usage.output = u.output_tokens || 0;
    msg.usage.cacheRead = (u.input_tokens_details && u.input_tokens_details.cached_tokens) || 0;
    msg.usage.totalTokens = u.total_tokens || 0;
    this.contextTokens = msg.usage.totalTokens;
    if (msg.content.some((b) => b.type === 'toolCall')) msg.stopReason = 'toolUse';
    else if (final.status === 'incomplete') msg.stopReason = 'length';
    return msg;
  }

  async runTool(call) {
    try {
      const r = await fetch('/tool', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: call.name, arguments: call.arguments || {} }), signal: this.controller.signal,
      });
      const out = await r.json();
      if (out.error) return { text: out.error, isError: true };
      if (call.name === 'web_search') {
        const rs = out.results || [];
        return { text: rs.length ? rs.map((x, i) => `${i + 1}. ${x.title}\n   ${x.url}\n   ${x.snippet}`).join('\n\n') : 'No results', isError: false };
      }
      if (call.name === 'fetch_page') return { text: `${out.url}${out.truncated ? ' (first 8000 characters)' : ''}\n\n${out.text}`, isError: false };
      if (call.name === 'bash') return { text: out.output, isError: out.exitCode !== 0 };
      return { text: JSON.stringify(out), isError: false };
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      return { text: e.message, isError: true };
    }
  }
}

function textOf(m) {
  if (typeof m.content === 'string') return m.content;
  return (m.content || []).filter((b) => b.type === 'text').map((b) => b.text).join('');
}

// Arguments while they stream: parse what has arrived, closing open strings and objects.
function partialJson(raw) {
  try { return JSON.parse(raw); } catch (e) { /* incomplete */ }
  let s = raw, inStr = false, esc = false;
  const stack = [];
  for (const c of s) {
    if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; }
    else if (c === '"') inStr = true;
    else if (c === '{' || c === '[') stack.push(c === '{' ? '}' : ']');
    else if (c === '}' || c === ']') stack.pop();
  }
  if (inStr) s += '"';
  s += stack.reverse().join('');
  try { return JSON.parse(s); } catch (e) { return {}; }
}

export async function createTransport() { return new TrashcanTransport(); }
