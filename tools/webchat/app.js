// The chat: keeps the conversation as Responses input items, streams each round from the trashcan server (/v1 is
// passed through by server.py), runs the tools the model calls through server.py's /tool, and repeats until the
// model answers. Thinking and tool calls go in an expanded "Working" section above each answer.
const INSTRUCTIONS = `You are a helpful assistant running on a cluster of six 2013 Mac Pros.
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
];
const MAX_ROUNDS = 8;

const $ = (id) => document.getElementById(id);
const stream = $('stream'), input = $('input'), statusEl = $('status');
let model = null, history = [], controller = null;

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function pinned() { return stream.scrollHeight - stream.scrollTop - stream.clientHeight < 80; }
function scroll(was) { if (was) stream.scrollTop = stream.scrollHeight; }
function setStatus(s) { statusEl.textContent = s; }

async function loadModel() {
  try {
    const r = await fetch('/v1/models');
    const j = await r.json();
    model = j.data && j.data[0] && j.data[0].id;
    $('title').textContent = model ? `trashcan - ${model}` : 'trashcan';
    setStatus(model ? 'Ready' : 'No model');
  } catch (e) {
    setStatus('No trashcan server');
  }
}

// One server round: stream events, build the UI, return the response's output items.
async function round(turn) {
  const body = {
    model, instructions: INSTRUCTIONS, input: history, tools: TOOLS, stream: true, store: false,
    reasoning: { effort: 'medium', summary: 'auto' }, max_output_tokens: 8192,
  };
  const r = await fetch('/v1/responses', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal,
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  const items = new Map();   // item_id -> {type, el, text}
  let final = null;
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let k;
    while ((k = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, k); buf = buf.slice(k + 2);
      const data = chunk.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).join('');
      if (!data || data === '[DONE]') continue;
      let ev;
      try { ev = JSON.parse(data); } catch (e) { continue; }
      const was = pinned();
      handle(ev, items, turn);
      if (ev.type === 'response.completed' || ev.type === 'response.incomplete') final = ev.response;
      if (ev.type === 'response.failed') throw new Error((ev.response && ev.response.error && ev.response.error.message) || 'generation failed');
      scroll(was);
    }
  }
  if (!final) throw new Error('the response ended early');
  if (final.status === 'incomplete') setStatus('Stopped at the output limit');
  return final.output || [];
}

function handle(ev, items, turn) {
  const t = ev.type;
  if (t === 'response.output_item.added') {
    const it = ev.item || {};
    if (it.type === 'reasoning') {
      const d = el('details', 'think', '<summary>Thinking</summary><div class="body md"></div>');
      d.open = true;
      turn.work.appendChild(d);
      items.set(it.id, { el: d, text: '' });
      turn.showWork();
    } else if (it.type === 'message') {
      const m = el('div', 'answer md');
      turn.box.appendChild(m);
      items.set(it.id, { el: m, text: '' });
    } else if (it.type === 'function_call') {
      const d = el('details', 'tool', `<summary>${esc(it.name || 'tool')}</summary><div class="args"></div><div class="result">Running</div>`);
      d.open = true;
      turn.work.appendChild(d);
      items.set(it.id, { el: d, text: '' });
      turn.showWork();
    }
  } else if (t === 'response.reasoning_summary_text.delta') {
    const b = items.get(ev.item_id);
    if (b) { b.text += ev.delta || ''; b.el.querySelector('.body').innerHTML = renderMarkdown(b.text); }
  } else if (t === 'response.output_text.delta') {
    const b = items.get(ev.item_id);
    if (b) { b.text += ev.delta || ''; b.el.innerHTML = renderMarkdown(b.text); }
  } else if (t === 'response.function_call_arguments.delta') {
    const b = items.get(ev.item_id);
    if (b) { b.text += ev.delta || ''; b.el.querySelector('.args').textContent = b.text; }
  } else if (t === 'response.output_item.done') {
    const it = ev.item || {};
    const b = items.get(it.id);
    if (b && it.type === 'function_call') {
      b.el.querySelector('summary').textContent = `${it.name}: ${summarizeArgs(it.arguments)}`;
      b.el.querySelector('.args').textContent = '';
      turn.calls.set(it.call_id, b.el);
    }
  }
}

function summarizeArgs(a) {
  try { const j = JSON.parse(a || '{}'); return j.query || j.url || JSON.stringify(j); } catch (e) { return a; }
}

async function runTool(item, card) {
  let args = {};
  try { args = JSON.parse(item.arguments || '{}'); } catch (e) { /* the server reports it */ }
  const r = await fetch('/tool', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: item.name, arguments: args }), signal: controller.signal,
  });
  const out = await r.json();
  if (card) card.querySelector('.result').innerHTML = toolResultHtml(item.name, out);
  return JSON.stringify(out);
}

function toolResultHtml(name, out) {
  if (out.error) return `<span class="error">${esc(out.error)}</span>`;
  if (name === 'web_search') {
    if (!out.results.length) return 'No results';
    return '<ol>' + out.results.map((x) => `<li><a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.title || x.url)}</a><br><span class="snippet">${esc(x.snippet)}</span></li>`).join('') + '</ol>';
  }
  if (name === 'fetch_page') return `${esc(out.text.length)} characters from <a href="${esc(out.url)}" target="_blank" rel="noopener">${esc(out.url)}</a>`;
  return esc(JSON.stringify(out).slice(0, 500));
}

async function send(text) {
  const user = el('div', 'user', esc(text).replace(/\n/g, '<br>'));
  stream.appendChild(user);
  const box = el('div', 'turn');
  const workBox = el('details', 'work', '<summary>Working</summary><div class="items"></div>');
  workBox.open = true;
  workBox.hidden = true;
  box.appendChild(workBox);
  stream.appendChild(box);
  stream.scrollTop = stream.scrollHeight;
  const turn = { box, work: workBox.querySelector('.items'), calls: new Map(), showWork: () => { workBox.hidden = false; } };

  history.push({ role: 'user', content: [{ type: 'input_text', text }] });
  controller = new AbortController();
  $('send').disabled = true; $('stop').disabled = false;
  const started = performance.now();
  try {
    for (let n = 0; n < MAX_ROUNDS; n++) {
      setStatus(n ? 'Working' : 'Thinking');
      const output = await round(turn);
      history.push(...output);
      const calls = output.filter((it) => it.type === 'function_call');
      if (!calls.length) break;
      setStatus('Searching');
      const results = await Promise.all(calls.map((c) => runTool(c, turn.calls.get(c.call_id))));
      calls.forEach((c, i) => history.push({ type: 'function_call_output', call_id: c.call_id, output: results[i] }));
    }
    setStatus(`Done in ${((performance.now() - started) / 1000).toFixed(1)} s`);
  } catch (e) {
    if (e.name === 'AbortError') setStatus('Stopped');
    else { box.appendChild(el('div', 'error', esc(e.message))); setStatus('Error'); }
    // Keep the history consistent: drop a trailing unanswered tool call round.
    while (history.length && history[history.length - 1].type === 'function_call') history.pop();
  } finally {
    controller = null;
    $('send').disabled = false; $('stop').disabled = true;
    input.focus();
  }
}

$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || controller) return;
  input.value = '';
  send(text);
});
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('composer').requestSubmit(); }
});
$('stop').addEventListener('click', () => controller && controller.abort());
$('new').addEventListener('click', () => {
  if (controller) controller.abort();
  history = [];
  stream.innerHTML = '';
  setStatus(model ? 'Ready' : '');
  input.focus();
});
loadModel();
