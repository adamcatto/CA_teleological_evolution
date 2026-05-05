/* global fetch */

const $ = (id) => document.getElementById(id);

const state = {
  apiBase: "",
  session: "",
  tokens: [],
  colorById: new Map(),
  lastStream: null,
  currentEpisode: null,
  pollTimer: null,
  metricsLoading: false,
  /** History dropdown: show last N episodes (must match server manifest / --viewer-max-episodes) */
  maxEpisodeFiles: 10000,
};

/** Root-absolute API URL so fetches work with any page path; override with ?api=http://host:port */
function apiUrl(pathWithLeadingSlash) {
  const r = pathWithLeadingSlash.startsWith("/") ? pathWithLeadingSlash : "/" + pathWithLeadingSlash;
  if (state.apiBase && state.apiBase.length > 0) {
    return state.apiBase.replace(/\/$/, "") + r;
  }
  if (
    typeof window !== "undefined" &&
    window.location &&
    window.location.origin &&
    window.location.origin !== "null"
  ) {
    return new URL(r, window.location.origin).href;
  }
  return r;
}

function rgbCss(rgb) {
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status + " " + (await r.text()));
  return r.json();
}

const CHART_REWARD = "#4caf50";
const CHART_LOSS = "#e57373";

/**
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} steps
 * @param {number[]} values
 * @param {string} color
 */
function drawMetricLineChart(canvas, steps, values, color) {
  const ctx = canvas.getContext("2d");
  const w = Math.max(120, Math.floor(canvas.getBoundingClientRect().width) || 320);
  const h = 88;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.height = h + "px";
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const padL = 38;
  const padR = 6;
  const padT = 6;
  const padB = 14;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const n = steps.length;
  if (n < 2) {
    ctx.fillStyle = "#7d8288";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(n === 0 ? "No data yet" : "Need 2+ points", padL, h / 2);
    return;
  }
  const pairs = [];
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || !Number.isFinite(steps[i])) continue;
    pairs.push({ s: steps[i], v });
  }
  if (pairs.length < 2) {
    ctx.fillStyle = "#7d8288";
    ctx.fillText("No finite y values", padL, h / 2);
    return;
  }
  const xs0 = pairs[0].s;
  const xs1 = pairs[pairs.length - 1].s;
  const yVals = pairs.map((p) => p.v);
  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  if (yMax === yMin) {
    yMin -= 1;
    yMax += 1;
  }
  const xSpan = Math.max(1e-9, xs1 - xs0);
  const ySpan = Math.max(1e-9, yMax - yMin);
  const xToPx = (s) => padL + ((s - xs0) / xSpan) * innerW;
  const yToPx = (v) => padT + innerH - ((v - yMin) / ySpan) * innerH;
  ctx.strokeStyle = "#2a2f3a";
  ctx.beginPath();
  ctx.moveTo(padL, padT + innerH);
  ctx.lineTo(padL + innerW, padT + innerH);
  ctx.stroke();
  ctx.fillStyle = "#7d8288";
  ctx.font = "9px ui-monospace, Menlo, monospace";
  ctx.fillText(String(yMin.toPrecision(4)), 2, padT + innerH);
  ctx.fillText(String(yMax.toPrecision(4)), 2, padT + 9);
  ctx.textAlign = "left";
  ctx.fillText("step " + String(xs0), padL, h - 3);
  ctx.textAlign = "right";
  ctx.fillText(String(xs1), w - 4, h - 3);
  ctx.textAlign = "left";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let i = 0; i < pairs.length; i++) {
    const x = xToPx(pairs[i].s);
    const y = yToPx(pairs[i].v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

/**
 * @param {any} data stream.json payload (may include `metrics_points` from viewer_server)
 */
function buildMetricsViewFromStreamData(data) {
  const pts = (data && data.metrics_points) || [];
  return {
    points: pts,
    total_rows: data.metrics_total_rows != null ? data.metrics_total_rows : pts.length,
    n: pts.length,
    path: data.metrics_path || "",
    error: data.metrics_error || "",
    fromStream: true,
  };
}

async function loadAndDrawMetrics() {
  const se = state.session;
  if (!se || ! $("showMetricsCharts")?.checked) return;
  const status = $("metricsChartStatus");
  if (status) status.textContent = "Loading…";
  if (state.metricsLoading) return;
  state.metricsLoading = true;
  try {
    let m;
    if (state.lastStream && state.lastStream.metrics_points && state.lastStream.metrics_points.length) {
      m = buildMetricsViewFromStreamData(state.lastStream);
    } else {
      try {
        m = await fetchJson(apiUrl("/api/metrics?session=" + encodeURIComponent(se)));
        m = { points: m.points || [], total_rows: m.total_rows, n: m.n, path: m.path, error: m.error || "" };
      } catch (e) {
        try {
          const sd = await fetchJson(
            apiUrl("/api/stream?session=" + encodeURIComponent(se) + "&stream_metrics=1&stream_metrics_max=20000")
          );
          if (state.lastStream) {
            Object.assign(state.lastStream, sd);
          } else {
            state.lastStream = sd;
          }
          m = buildMetricsViewFromStreamData(sd);
          if (!m.points || !m.points.length) {
            throw e;
          }
        } catch (e2) {
          throw e;
        }
      }
    }
    if (!m.points) {
      m = { points: [], total_rows: 0, n: 0, error: m.error || "" };
    }
    const points = m.points;
    const st = [];
    const r = [];
    const l = [];
    for (const p of points) {
      const s = p.global_step;
      if (!Number.isFinite(s)) continue;
      st.push(s);
      r.push(p.reward);
      l.push(p.loss);
    }
    if (status) {
      const err = m.error || "";
      const tr = m.total_rows != null ? m.total_rows : points.length;
      const src = m.fromStream ? " (from stream)" : "";
      status.textContent =
        tr +
        " rows" +
        (points.length < tr ? " (showing last " + points.length + ")" : "") +
        src +
        (err ? " — " + err : "");
    }
    drawMetricLineChart($("chartReward"), st, r, CHART_REWARD);
    drawMetricLineChart($("chartLoss"), st, l, CHART_LOSS);
  } catch (e) {
    if (status) status.textContent = "Error: " + e;
  } finally {
    state.metricsLoading = false;
  }
}

function buildColorMap() {
  state.colorById = new Map();
  for (const t of state.tokens) {
    state.colorById.set(t.id, t.rgb);
  }
}

function colorForToken(id) {
  return state.colorById.get(id) || [50, 50, 50];
}

function drawTokenCanvas(canvas, grid) {
  const ctx = canvas.getContext("2d");
  if (!grid || !grid.length) {
    return;
  }
  const h = grid.length;
  const w = grid[0].length;
  const px = $pixelSize(w, h);
  canvas.width = w * px;
  canvas.height = h * px;
  const img = ctx.createImageData(canvas.width, canvas.height);
  const data = img.data;
  let p = 0;
  for (let y = 0; y < h; y++) {
    for (let yy = 0; yy < px; yy++) {
      for (let x = 0; x < w; x++) {
        const c = colorForToken(grid[y][x]);
        for (let xx = 0; xx < px; xx++) {
          const i = (p + x * px + xx) * 4;
          data[i] = c[0];
          data[i + 1] = c[1];
          data[i + 2] = c[2];
          data[i + 3] = 255;
        }
      }
      p += canvas.width;
    }
  }
  ctx.putImageData(img, 0, 0);
  canvas.style.imageRendering = $("pixelated")?.checked ? "pixelated" : "auto";
}

let $maxPx = 6;
function $pixelSize(w, h) {
  const maxDim = Math.max(w, h);
  if (maxDim > 256) return 2;
  if (maxDim > 128) return 3;
  if (maxDim > 64) return 4;
  return $maxPx;
}

function drawGridMain(grid) {
  drawTokenCanvas($("gridCanvas"), grid);
  const el = $("gridTitle");
  if (grid && grid[0]) el.textContent = `— ${grid.length}×${grid[0].length}`;
  else el.textContent = "";
  renderCellCounts(grid);
}

/** Per–vocab-token counts for the current frame (left pane). */
function renderCellCounts(grid) {
  const host = $("cellCounts");
  if (!host) return;
  if (!grid || !grid.length || !state.tokens.length) {
    host.textContent = "";
    return;
  }
  const h = grid.length;
  const w = grid[0].length;
  const byId = new Map();
  for (const t of state.tokens) byId.set(t.id, 0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const id = grid[y][x];
      byId.set(id, (byId.get(id) || 0) + 1);
    }
  }
  host.innerHTML = "";
  const total = h * w;
  const tot = document.createElement("div");
  tot.className = "cell-counts-total";
  tot.textContent = "Total: " + total;
  host.appendChild(tot);
  for (const t of state.tokens) {
    const n = byId.get(t.id) ?? 0;
    const row = document.createElement("div");
    row.className = "count-row";
    const sw = document.createElement("span");
    sw.className = "count-swatch";
    sw.style.background = rgbCss(t.rgb);
    const lab = document.createElement("span");
    lab.className = "count-name";
    lab.textContent = t.name;
    const num = document.createElement("span");
    num.className = "count-n";
    num.textContent = n;
    row.appendChild(sw);
    row.appendChild(lab);
    row.appendChild(num);
    host.appendChild(row);
  }
}

function renderLegend() {
  const leg = $("legend");
  leg.innerHTML = "";
  for (const t of state.tokens) {
    const d = document.createElement("span");
    d.className = "leg";
    const sw = document.createElement("i");
    sw.style.background = rgbCss(t.rgb);
    d.appendChild(sw);
    d.appendChild(document.createTextNode(t.name + " (" + t.id + ")"));
    leg.appendChild(d);
  }
}

function setMeta(html) {
  $("runMeta").innerHTML = html;
}

/**
 * @param {number|string|null|undefined} stage  null/undefined → em dash; stage 0 is valid.
 */
function setCurriculumStageDisplay(stage) {
  const el = $("curriculumStage");
  if (!el) return;
  if (stage == null) {
    el.textContent = "—";
    return;
  }
  if (typeof stage === "number" && Number.isFinite(stage)) {
    el.textContent = "Stage " + stage;
    return;
  }
  const n = Number(stage);
  if (Number.isFinite(n)) {
    el.textContent = "Stage " + n;
  } else {
    el.textContent = String(stage);
  }
}

/** Live stream: prefer top-level ``curriculum_stage`` (see ``viewer_log``), then ``latest_episode.extra``. */
function setCurriculumStageFromStreamData(data) {
  const le = data && data.latest_episode;
  const s = data?.curriculum_stage ?? le?.extra?.curriculum_stage;
  setCurriculumStageDisplay(s);
}

function renderRules(rules) {
  const host = $("rulesList");
  host.innerHTML = "";
  if (!rules || !rules.length) {
    const p = document.createElement("p");
    p.className = "meta";
    p.textContent = "No rules (grid pretrain or empty rule set).";
    host.appendChild(p);
    return;
  }
  const cap = Math.min(rules.length, 256);
  for (let i = 0; i < cap; i++) {
    const r = rules[i];
    const block = document.createElement("div");
    block.className = "rule-block";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = (r.name || "rule") + " [" + (i + 1) + "]";
    block.appendChild(name);
    const row = document.createElement("div");
    row.className = "rule-row";
    row.appendChild(mini3(r.input));
    row.appendChild(document.createTextNode("→"));
    row.appendChild(mini3(r.output));
    block.appendChild(row);
    host.appendChild(block);
  }
  if (rules.length > cap) {
    const p = document.createElement("p");
    p.className = "meta";
    p.textContent = "… and " + (rules.length - cap) + " more (not shown)";
    host.appendChild(p);
  }
}

function mini3(cells) {
  const w = document.createElement("div");
  w.className = "mini3";
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < 3; x++) {
      const s = document.createElement("span");
      const id = cells[y][x];
      s.style.background = rgbCss(colorForToken(id));
      s.title = String(id);
      w.appendChild(s);
    }
  }
  return w;
}

function setSlider(frames) {
  const n = (frames && frames.length) || 0;
  const s = $("timeSlider");
  s.min = 0;
  s.max = Math.max(0, n - 1);
  s.value = Math.min(n - 1, Math.max(0, n - 1));
  $("timeLabel").textContent = n ? "0 / " + (n - 1) : "0 / 0";
  if (n) showFrame(frames, 0);
  s.oninput = () => {
    const i = +s.value;
    $("timeLabel").textContent = i + " / " + (n - 1);
    showFrame(frames, i);
  };
}

function showFrame(frames, i) {
  if (!frames || !frames[i]) return;
  drawGridMain(frames[i]);
}

async function loadEpisodes() {
  const se = $("sessionSelect").value || $("sessionManual").value.trim() || state.session;
  if (!se) return;
  let cap = 10000;
  try {
    const meta = await fetchJson(apiUrl("/api/meta?session=" + encodeURIComponent(se)));
    if (meta && meta.max_episode_files != null && Number.isFinite(+meta.max_episode_files)) {
      cap = Math.max(1, +meta.max_episode_files);
    }
  } catch (e) {
    /* default cap */
  }
  state.maxEpisodeFiles = cap;
  const m = await fetchJson(apiUrl("/api/episodes?session=" + encodeURIComponent(se)));
  const items = m.items || [];
  const es = $("episodeSelect");
  es.innerHTML = "";
  for (const it of items.slice(-cap).reverse()) {
    const o = document.createElement("option");
    o.value = it.file;
    o.textContent = "id " + it.id + "  " + (it.phase || "") + "  r=" + (it.reward != null ? it.reward.toFixed(3) : "?");
    es.appendChild(o);
  }
  es.onchange = async () => {
    const file = es.value;
    if (!file) return;
    const ep = await fetchJson(
      apiUrl("/api/episode?session=" + encodeURIComponent(se) + "&file=" + encodeURIComponent(file))
    );
    state.currentEpisode = ep;
    setCurriculumStageDisplay(ep?.extra?.curriculum_stage);
    setMeta(
      "phase " +
        (ep.phase || "") +
        "  reward " +
        (ep.reward != null ? ep.reward : "") +
        "  id " +
        ep.id +
        "  frames " +
        (ep.n_frames || (ep.frames && ep.frames.length)) +
        (ep.extra ? "\n" + JSON.stringify(ep.extra) : "")
    );
    renderRules(ep.rules);
    setSlider(ep.frames);
  };
  if (es.options.length) es.dispatchEvent(new Event("change"));
  else setCurriculumStageDisplay(null);
}

function stopPoll() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function pickLiveGrid(data) {
  const phase = data.phase;
  const pre = data.pretrain || {};
  const best = pre.best_in_window;
  const useBest = phase === "pretrain_grid" && $("pretrainShowBest")?.checked && best && best.grid;
  if (useBest) return { grid: best.grid, label: "pretrain best-in-window  r=" + (best.reward != null ? best.reward.toFixed(4) : "") };
  const le = data.latest_episode;
  const full = data._fullEpisode;
  if (full && full.frames && full.frames.length) {
    return { grid: full.frames[full.frames.length - 1], label: "latest episode last frame" };
  }
  if (le && le.last_frame) {
    return { grid: le.last_frame, label: "latest last_frame (stream)" };
  }
  return { grid: null, label: "" };
}

function applyStreamToUi(data) {
  state.lastStream = data;
  const le = data.latest_episode;
  if (!le) {
    setCurriculumStageDisplay(null);
    setMeta("waiting for training…");
    return;
  }
  setCurriculumStageFromStreamData(data);
  const full = data._fullEpisode;
  const { grid, label } = pickLiveGrid(data);
  let meta =
    "global_step " +
    data.global_step +
    "  phase " +
    (data.phase || "") +
    (label ? "  " + label : "") +
    (le && le.reward != null ? "  ep_reward " + le.reward : "") +
    (le && le.solved != null ? "  solved " + le.solved : "") +
    (le.file ? "  file " + le.file : "");
  if (data._episodeError) meta += "\nepisode fetch: " + data._episodeError;
  if (le && le.extra) meta += "\n" + JSON.stringify(le.extra);
  setMeta(meta);
  if ($("sourceMode").value === "history") return;
  if (full && full.rules) renderRules(full.rules);
  else renderRules(le.rules);
  if (grid) {
    setSlider([grid]);
  } else if (full && full.frames) {
    setSlider(full.frames);
  } else if (le.last_frame) {
    setSlider([le.last_frame]);
  }
}

async function pollStream() {
  const se = state.session;
  if (!se) return;
  try {
    const data = await fetchJson(apiUrl("/api/stream?session=" + encodeURIComponent(se)));
    const le = data.latest_episode;
    if (le && le.file && (le.n_frames || 0) > 1) {
      try {
        const full = await fetchJson(
          apiUrl(
            "/api/episode?session=" +
              encodeURIComponent(se) +
              "&file=" +
              encodeURIComponent(le.file)
          )
        );
        data._fullEpisode = full;
      } catch (e) {
        data._episodeError = String(e);
      }
    }
    if ($("sourceMode").value === "live") applyStreamToUi(data);
    if ($("sourceMode").value === "live" && $("showMetricsCharts")?.checked) {
      void loadAndDrawMetrics();
    }
  } catch (e) {
    setMeta("poll error: " + e);
  }
}

function startPoll() {
  stopPoll();
  if ($("sourceMode").value !== "live") return;
  const ms = +($("pollMs")?.value || 1200);
  state.pollTimer = setInterval(pollStream, Math.max(200, ms));
  pollStream();
}

function onSourceChange() {
  const live = $("sourceMode").value === "live";
  $("historyControls").hidden = live;
  $("liveOptions").style.display = live ? "block" : "none";
  if (live) startPoll();
  else stopPoll();
}

async function init() {
  const sp = new URLSearchParams(location.search);
  const qs = sp.get("session");
  const apiOverride = sp.get("api");
  if (apiOverride) state.apiBase = apiOverride.replace(/\/$/, "");
  if (qs) {
    state.session = qs;
    const m = $("sessionManual");
    if (m) m.value = qs;
  }
  const v = await fetchJson(apiUrl("/api/vocab"));
  state.tokens = v.tokens || [];
  buildColorMap();
  renderLegend();

  const list = await fetchJson(apiUrl("/api/sessions"));
  const ss = $("sessionSelect");
  ss.innerHTML = '<option value="">(pick)</option>';
  for (const s of list.sessions || []) {
    const o = document.createElement("option");
    o.value = s.session;
    o.textContent = s.session;
    if (s.mtime) o.title = new Date(s.mtime * 1000).toISOString();
    ss.appendChild(o);
  }
  if (ss.options.length > 1) {
    ss.selectedIndex = 1;
    state.session = ss.value;
  }

  ss.onchange = () => {
    state.session = ss.value;
    $("sessionManual").value = state.session;
    onSourceChange();
    loadEpisodes();
    if ($("showMetricsCharts")?.checked) void loadAndDrawMetrics();
  };
  $("applySession").onclick = () => {
    const m = $("sessionManual").value.trim();
    if (m) state.session = m;
    for (const o of ss.options) {
      if (o.value === state.session) {
        o.selected = true;
        break;
      }
    }
    onSourceChange();
    loadEpisodes();
    if ($("showMetricsCharts")?.checked) void loadAndDrawMetrics();
  };

  $("sourceMode").onchange = onSourceChange;
  const metricsCb = $("showMetricsCharts");
  if (metricsCb) {
    metricsCb.onchange = () => {
      const c = metricsCb.checked;
      const wrap = $("metricsChartsWrap");
      if (wrap) wrap.hidden = !c;
      if (c) void loadAndDrawMetrics();
    };
  }
  $("pretrainShowBest").onchange = () => {
    if (state.lastStream) applyStreamToUi(state.lastStream);
  };
  $("pollMs").onchange = startPoll;
  $("pixelated").onchange = () => {
    const g = state.currentEpisode;
    if (g && g.frames) showFrame(g.frames, +$("timeSlider").value);
    else if (state.lastStream && state.lastStream.latest_episode) {
      const le = state.lastStream.latest_episode;
      if (le.frames) showFrame(le.frames, +$("timeSlider").value);
    }
  };

  onSourceChange();
  if ($("sourceMode").value === "history") loadEpisodes();
}

document.addEventListener("DOMContentLoaded", init);
