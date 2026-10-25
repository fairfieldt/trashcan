// Horizontal bar charts drawn at the container's real width, so text stays legible on phones.
// Every chart reads its numbers from the CHARTS table below; the same numbers appear in each
// figure's data table and in notes.html.

const NS = "http://www.w3.org/2000/svg";

const CHARTS = {
  quality: {
    unit: "%", max: 100, ticks: [0, 20, 40, 60, 80, 100],
    rows: [
      { label: "Six-Mac recipe (IQ4 experts, Q6_K head)", segs: [{ v: 95.1, cls: "b1", tip: "95.1% same top token; mean KLD 0.0391" }], total: "95.1%" },
      { label: "Q4_0 experts", segs: [{ v: 94.9, cls: "bsoft", tip: "94.9%; mean KLD 0.0402" }], total: "94.9%" },
      { label: "MXFP4 experts", segs: [{ v: 93.7, cls: "bsoft", tip: "93.7%; mean KLD 0.0656" }], total: "93.7%" },
      { label: "IQ3_XXS experts", segs: [{ v: 92.6, cls: "bsoft", tip: "92.6%; mean KLD 0.0876" }], total: "92.6%" },
      { label: "IQ2_XXS experts", segs: [{ v: 88.6, cls: "bsoft", tip: "88.6%; mean KLD 0.2134" }], total: "88.6%" },
    ],
  },
  weights: {
    unit: "GiB", max: 120, ticks: [0, 20, 40, 60, 80, 100, 120],
    rows: [
      { label: "Model weights (G-iq4 recipe)", segs: [
        { v: 65, cls: "b1", name: "In GPU memory", tip: "65 GiB must sit in GPU memory (116 − 51, calculated)" },
        { v: 51, cls: "b2", name: "N-gram table, on an SSD", tip: "51 GiB n-gram table, read from the first Mac's SSD" },
      ], total: "116 GiB" },
      { label: "GPU memory, 12 × D700", segs: [
        { v: 72, cls: "bsoft", name: "GPU memory", tip: "12 cards × 6 GiB = 72 GiB, as twelve separate 6 GiB pools" },
      ], total: "72 GiB" },
    ],
  },
  vram: {
    unit: "MiB", max: 6144, ticks: [0, 1024, 2048, 3072, 4096, 5120, 6144],
    refs: [{ x: 5869, label: "budget 5,869" }],
    rows: [
      { label: "incite GPU 0: layers 0–3, n-gram module, 1/12 of the head", segs: [
        { v: 5830, cls: "b1", name: "Allocated", tip: "5,830 MiB allocated (measured). 39 MiB under the driver budget." },
      ], total: "5,830" },
    ],
  },
  decode: {
    unit: "tok/s", max: 50, ticks: [0, 10, 20, 30, 40, 50],
    refs: [
      { x: 31, label: "≈31" },
      { x: 47, label: "47" },
    ],
    rows: [
      { label: "Oct 12 · first run, automatic GPU clocks", segs: [{ v: 8.1, cls: "b1", tip: "8.0–8.2 tok/s. amdgpu kept dropping clocks between bursts." }], total: "8.1" },
      { label: "Oct 12 · GPUs held at high clock", segs: [{ v: 13.5, cls: "b1", tip: "13.5 tok/s" }], total: "13.5" },
      { label: "Oct 12 · every Mac scores its head slice at once", segs: [{ v: 14.7, cls: "b1", tip: "13.53 → 14.70 tok/s" }], total: "14.7" },
      { label: "Oct 12 · decode kernels, round 1", segs: [{ v: 17.9, cls: "b1", tip: "14.70 → 17.92 tok/s" }], total: "17.9" },
      { label: "Oct 13 · small-batch expert kernels", segs: [{ v: 19.0, cls: "b1", tip: "18.13 → 19.04 tok/s" }], total: "19.0" },
      { label: "Oct 13 · 1–4-token kernels, long-context round", segs: [{ v: 20.6, cls: "b1", tip: "20.67, 20.51 tok/s (main e9f34e6)" }], total: "20.6" },
      { label: "Oct 14 · host path between Macs", segs: [{ v: 21.1, cls: "b1", tip: "21.09, 21.20 tok/s (main 15e5457)" }], total: "21.1" },
      { label: "Oct 14 · Linux 7.3, USB4STREAM, decode kernels round 2", segs: [{ v: 22.5, cls: "b1", tip: "22.52, 22.54 tok/s (main 4f36bf0)" }], total: "22.5" },
    ],
  },
  prefill: {
    unit: "tok/s", max: 1400, ticks: [0, 200, 400, 600, 800, 1000, 1200, 1400],
    rows: [
      { label: "512-token prompt", segs: [{ v: 523.1, cls: "b1", tip: "522.9, 523.3 tok/s: about 1.0 s" }], total: "523" },
      { label: "2,048 tokens", segs: [{ v: 889.8, cls: "b1", tip: "889.8 tok/s: about 2.3 s" }], total: "890" },
      { label: "8,192 tokens", segs: [{ v: 1118.0, cls: "b1", tip: "1,118.0 tok/s: about 7.3 s" }], total: "1,118" },
      { label: "30,000 tokens", segs: [{ v: 1249.3, cls: "b1", tip: "1,249.3 tok/s: about 24 s" }], total: "1,249" },
    ],
  },
  power: {
    unit: "W", max: 1600, ticks: [0, 400, 800, 1200, 1600],
    rows: [
      { label: "Decoding at 20.61 tok/s (six Macs, wall power)", segs: [
        { v: 732.5, cls: "b1", name: "12 GPUs", tip: "GPUs: 730–735 W (measured)" },
        { v: 341.5, cls: "b2", name: "6 CPU packages", tip: "CPU packages: 340–343 W (measured)" },
        { v: 286.5, cls: "b3", name: "Everything else", tip: "Everything else: ≈287 W (total minus GPUs and CPUs, calculated)" },
      ], total: "1,357–1,364 W" },
      { label: "Waiting for a request, engine running", segs: [
        { v: 335, cls: "bghost", name: "", tip: "320–350 W for six Macs, idle cores busy-polling (measured)" },
      ], total: "320–350 W" },
    ],
  },
};

function el(name, attrs, parent) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function fmt(n) {
  return n.toLocaleString("en-US");
}

function draw(host) {
  const spec = CHARTS[host.dataset.chart];
  const W = Math.max(280, host.clientWidth);
  const padL = 2;
  const padR = Math.max(...spec.rows.map(r => r.total.length)) * 7.6 + 12;
  const plotW = W - padL - padR;
  const charW = 7.1;
  const wrap = text => {
    const max = Math.max(20, Math.floor((W - padL) / charW));
    const lines = [];
    let cur = "";
    for (const w of text.split(" ")) {
      if (cur && (cur + " " + w).length > max) { lines.push(cur); cur = w; } else cur = cur ? cur + " " + w : w;
    }
    lines.push(cur);
    return lines;
  };
  const labels = spec.rows.map(r => wrap(r.label));
  const top = spec.refs ? 24 : 8;
  const rowHs = labels.map(l => 34 + l.length * 17);
  const H = top + rowHs.reduce((a, b) => a + b, 0) + 26;
  const x = v => padL + (v / spec.max) * plotW;

  host.textContent = "";
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": host.dataset.label || "" }, host);

  const step = plotW / (spec.ticks.length - 1) < 58 ? 2 : 1;
  spec.ticks.forEach((t, i) => {
    el("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 4, y2: H - 22 }, svg);
    const last = i === spec.ticks.length - 1;
    if ((i % step && !last) || (!last && x(spec.max) - x(t) < 70 && i > 0)) return;
    const tx = el("text", { class: "tick", x: x(t), y: H - 6, "text-anchor": i === 0 ? "start" : last ? "end" : "middle" }, svg);
    tx.textContent = fmt(t) + (last ? " " + spec.unit : "");
  });

  let y0 = top;
  spec.rows.forEach((row, i) => {
    const lbl = el("text", { class: "lbl", x: padL, y: y0 + 14 }, svg);
    labels[i].forEach((line, k) => {
      const ts = el("tspan", { x: padL, dy: k ? 17 : 0 }, lbl);
      ts.textContent = line;
    });
    let acc = 0;
    const barY = y0 + 5 + labels[i].length * 17, barH = 20;
    row.segs.forEach((s, j) => {
      const g = el("g", { class: "mark", tabindex: 0, "data-tip": s.tip, "aria-label": s.tip }, svg);
      const x0 = x(acc), x1 = x(acc + s.v);
      const gap = j < row.segs.length - 1 ? 2 : 0;
      el("rect", { class: s.cls, x: x0, y: barY, width: Math.max(1, x1 - x0 - gap), height: barH,
        rx: j === row.segs.length - 1 ? 3 : 0 }, g);
      el("rect", { x: x0, y: barY - 6, width: Math.max(8, x1 - x0), height: barH + 12, fill: "transparent" }, g);
      acc += s.v;
    });
    const v = el("text", { class: "val", x: x(acc) + 6, y: barY + 15 }, svg);
    v.textContent = row.total;
    y0 += rowHs[i];
  });

  (spec.refs || []).forEach(r => {
    el("line", { class: "ref", x1: x(r.x), x2: x(r.x), y1: 16, y2: H - 22 }, svg);
    const t = el("text", { class: "reflbl", x: x(r.x), y: 11, "text-anchor": "middle" }, svg);
    t.textContent = r.label;
  });
}

const tip = document.createElement("div");
tip.className = "tip";
tip.setAttribute("aria-hidden", "true");
document.body.appendChild(tip);

function showTip(target, cx, cy) {
  tip.textContent = target.dataset.tip;
  tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  let left = Math.min(window.innerWidth - r.width - 8, Math.max(8, cx - r.width / 2));
  let topY = cy - r.height - 12;
  if (topY < 8) topY = cy + 16;
  tip.style.left = left + "px";
  tip.style.top = topY + "px";
}

document.addEventListener("pointermove", e => {
  const m = e.target.closest && e.target.closest("g.mark");
  if (m) showTip(m, e.clientX, e.clientY); else tip.classList.remove("on");
});
document.addEventListener("focusin", e => {
  const m = e.target.closest && e.target.closest("g.mark");
  if (m) { const b = m.getBoundingClientRect(); showTip(m, b.left + b.width / 2, b.top); }
});
document.addEventListener("focusout", () => tip.classList.remove("on"));

const hosts = [...document.querySelectorAll("[data-chart]")];
const redraw = () => hosts.forEach(draw);
redraw();
let pending;
window.addEventListener("resize", () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(redraw); });
