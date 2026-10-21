/* ==========================================================================
   MedWatch ICU — "Thermal ward"
   Data below is simulated (seeded, so every reload shows the same patients).
   Model integration is unchanged: POST /predict on the local model API.
   ========================================================================== */

const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const TAU = Math.PI * 2;
let calm = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
function setCalm(v) { calm = v; document.body.classList.toggle('calm', v); }

/* ─── HEAT: the only colour in the interface ─── */
const STOPS = [[0, [90, 162, 255]], [0.30, [90, 162, 255]], [0.36, [198, 124, 214]], [0.42, [255, 179, 71]], [0.58, [255, 179, 71]], [0.70, [255, 68, 56]], [0.93, [255, 68, 56]], [1, [255, 217, 199]]];
function heatRGB(s) {
  s = clamp(s, 0, 1);
  for (let i = 1; i < STOPS.length; i++) {
    if (s <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i], t = (s - a) / (b - a || 1);
      return [lerp(ca[0], cb[0], t), lerp(ca[1], cb[1], t), lerp(ca[2], cb[2], t)];
    }
  }
  return STOPS[STOPS.length - 1][1];
}
const rgba = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
const heatCss = (s, a = 1) => rgba(heatRGB(s), a);
const heatHex = s => '#' + heatRGB(s).map(v => (v | 0).toString(16).padStart(2, '0')).join('');
const sevColor = sev => heatCss(0.05 + sev * 0.85);
const riskHex = r => r === 'HIGH' ? '#ff4438' : r === 'MEDIUM' ? '#ffb347' : '#5aa2ff';
const fmtS = s => s >= 0.995 ? '0.99+' : s.toFixed(2);
const riskLabel = r => r === 'HIGH' ? 'High risk' : r === 'MEDIUM' ? 'Medium' : 'Stable';
const riskCls = r => r === 'HIGH' ? 'hot' : r === 'MEDIUM' ? 'amber' : 'cold';
const PAL = { cold: '#5aa2ff', amber: '#ffb347', hot: '#ff4438', bone: '#f2effc', mute: '#8381a1', grid: 'rgba(188,186,255,.10)' };

/* ─── DATA: the real time-series dataset, scored by the trained LSTM (see prepare_data.py) ─── */
const STAFF = [
  { id: 'DS', name: 'Dr. D. Sharma', role: 'Senior Doctor', color: '#4a6fd0' },
  { id: 'AK', name: 'Dr. A. Kapoor', role: 'Resident Doctor', color: '#b7791f' },
  { id: 'RP', name: 'Dr. R. Patel', role: 'ICU Specialist', color: '#8b5cc7' },
  { id: 'NM', name: 'Nurse N. Mehta', role: 'Head Nurse', color: '#1f8a8a' },
  { id: 'SK', name: 'Nurse S. Kumar', role: 'Staff Nurse', color: '#d1495b' },
  { id: 'VR', name: 'Dr. V. Rao', role: 'Cardiologist', color: '#2b7aa8' },
  { id: 'PG', name: 'Dr. P. Gupta', role: 'Pulmonologist', color: '#a0663c' },
  { id: 'JS', name: 'Nurse J. Singh', role: 'Staff Nurse', color: '#3d9a6b' },
];
let selectedEditor = STAFF[0];
let editHistory = [];

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

const DATA = window.ICU_DATA;
const WARDS = 8, STEPS = DATA.steps;
const MODEL_API_URL = 'http://127.0.0.1:5000/predict';
const MODEL_HEALTH_URL = 'http://127.0.0.1:5000/health';
const lastOf = a => a[a.length - 1];
function carryBack(arr) { // readings before the model had 10 hours of history have no score: show the first real one
  const first = arr.find(v => v !== null); return arr.map(v => v === null ? first : v);
}
const patients = DATA.patients.map(d => {
  const lstmS = carryBack(d.risk), lstm = lastOf(lstmS);
  const k = STEPS - DATA.seqLen; // index of the first of the last 10 readings in the 24-reading series
  const p = {
    id: `P-${String(d.id).padStart(3, '0')}`, age: d.age, sex: d.sex, split: d.split, admitted: d.adm, readings: d.n,
    hrS: d.hr, spo2S: d.sp, sbpS: d.sb, dbpS: d.db, rrS: d.rr, tempS: d.tp, wbcS: d.wb, lacS: d.lc, lstmS, known: d.risk.map(v => v !== null),
    lstm, risk: lstm > 0.65 ? 'HIGH' : lstm > 0.35 ? 'MEDIUM' : 'LOW',
    hr: Math.round(lastOf(d.hr)), spo2: lastOf(d.sp), sbp: Math.round(lastOf(d.sb)), dbp: Math.round(lastOf(d.db)), rr: Math.round(lastOf(d.rr)), temp: lastOf(d.tp),
    wbc: lastOf(d.wb), lactate: lastOf(d.lc), creatinine: lastOf(d.cr), platelets: Math.round(lastOf(d.pl)), hgb: lastOf(d.hg),
    glucose: Math.round(lastOf(d.gl)), sodium: Math.round(lastOf(d.na)), potassium: lastOf(d.k), bun: Math.round(lastOf(d.bn)),
    // the exact 10 readings x 17 features the model sees
    rec: Array.from({ length: DATA.seqLen }, (_, i) => ({
      Age: d.age, Gender: d.sex === 'M' ? 1 : 0, Heart_Rate: d.hr[k + i], Systolic_BP: d.sb[k + i], Diastolic_BP: d.db[k + i], SpO2: d.sp[k + i],
      Respiratory_Rate: d.rr[k + i], Temperature: d.tp[k + i], WBC: d.wb[k + i], Lactate: d.lc[k + i], Creatinine: d.cr[i], Platelets: d.pl[i],
      Hemoglobin: d.hg[i], Glucose: d.gl[i], Sodium: d.na[i], Potassium: d.k[i], BUN: d.bn[i]
    })),
    ward: 1, bed: 1, alertH: 0
  };
  // hours the score has stayed above the warning threshold
  let h = 0; for (let i = STEPS - 1; i >= 0 && lstmS[i] > 0.35; i--) h++; p.alertH = Math.max(h, 1);
  return p;
});
// The dataset has no ward or bed, so patients are spread over wards for display only (shuffled, seeded).
(function assignWards() {
  const order = patients.map((_, i) => i), r = mulberry32(77);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  order.forEach((pi, pos) => { patients[pi].ward = pos % WARDS + 1; patients[pi].bed = Math.floor(pos / WARDS) + 1; });
})();
const BEDS_MAX = Math.max(...patients.map(p => p.bed));

const scoreAt = (p, step) => step >= STEPS - 1 ? p.lstm : clamp(p.lstmS[step], 0, 1);
const riskFromScore = s => s > 0.65 ? 'HIGH' : s > 0.35 ? 'MEDIUM' : 'LOW';
const sorted = (arr, f) => [...arr].sort((a, b) => f(b) - f(a));
const getTopRisk = () => sorted(patients, p => p.lstm).slice(0, 10);

/* ─── MODEL API (integration unchanged) ─── */
function buildModelRecords(p) { return p.rec; } // the real last 10 readings, all 17 features
async function predictPatientRisk(p) {
  const response = await fetch(MODEL_API_URL, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id: p.id, records: buildModelRecords(p) })
  });
  if (!response.ok) throw new Error('Prediction API unavailable');
  return response.json();
}
function applyModelPrediction(p, result) {
  const score = Number(result.risk_probability);
  if (Number.isNaN(score)) return;
  p.lstm = Math.max(0, Math.min(1, score));
  p.risk = result.risk_status?.toUpperCase().includes('HIGH') ? 'HIGH' : riskFromScore(p.lstm);
  p.lstmS = [...p.lstmS.slice(0, -1), +fmtS(p.lstm)];
}
function afterScoresChanged() {
  updateCounts(); Radar.retarget(); updateWardSide();
  if ($('page-patients').classList.contains('active')) filterPatients();
  if ($('page-alerts').classList.contains('active')) renderAlerts();
}
async function refreshDetailPrediction(idx) {
  const p = patients[idx];
  if (!p || p._predictionPending) return;
  p._predictionPending = true;
  try {
    applyModelPrediction(p, await predictPatientRisk(p));
    if (dossierIdx === idx) { renderGauge(p, false); $('dz-meta').innerHTML = metaHTML(p); }
    afterScoresChanged();
  } catch (e) { /* keep existing values when the local model API is offline */ }
  finally { p._predictionPending = false; }
}
async function refreshModelPredictions() {
  const btn = $('refresh-btn'); btn.classList.remove('spin'); void btn.offsetWidth; btn.classList.add('spin');
  try {
    await Promise.all(getTopRisk().map(async p => applyModelPrediction(p, await predictPatientRisk(p))));
    afterScoresChanged();
    showToast('Model predictions updated');
  } catch (e) { showToast('Model API not running; showing dashboard values'); }
  checkModelHealth();
}
let healthRetry = 0, healthTimer = null;
async function checkModelHealth() {
  clearTimeout(healthTimer);
  const pill = $('model-status'), txt = $('model-status-text'), sys = $('sys-model-status'), hint = $('sys-model-hint');
  let state = 'off', label = 'Model API offline', detail = 'Offline';
  try {
    const r = await fetch(MODEL_HEALTH_URL, { cache: 'no-store', signal: AbortSignal.timeout(6000) }), j = await r.json();
    if (j.model_loaded) { state = 'live'; label = 'LSTM model live'; detail = 'Running · LSTM loaded'; }
    else { state = 'fallback'; label = 'Fallback scoring'; detail = 'Running · fallback score'; }
    pill.title = j.message || 'Model API is running'; if (hint) hint.textContent = j.message || 'http://127.0.0.1:5000';
  } catch (e) { pill.title = 'Start the model API with: python model_api.py'; if (hint) hint.textContent = 'Not reachable. Start it with: python model_api.py'; }
  pill.dataset.state = state; txt.textContent = label; if (sys) sys.textContent = detail;
  // The API can take 15-60 s to load TensorFlow, so keep checking until it is live.
  if (state === 'live') { if (healthRetry > 0) refreshModelPredictions(); healthRetry = 0; }
  else if (healthRetry < 20 && !$('app').hidden) { healthRetry++; healthTimer = setTimeout(checkModelHealth, 5000); }
}

/* ─── CHARTS ─── */
const chartInstances = {};
function destroyChart(id) { if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; } }
const thresholdPlugin = {
  id: 'thresholds',
  afterDatasetsDraw(chart, args, opts) {
    if (!opts || !opts.lines) return;
    const { ctx, chartArea: { left, right }, scales: { y } } = chart;
    opts.lines.forEach(l => {
      const yy = y.getPixelForValue(l.value);
      ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = l.color; ctx.globalAlpha = .8; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.fillStyle = l.color; ctx.font = '500 11px "Instrument Sans", system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.fillText(l.label, right - 4, yy - 5); ctx.restore();
    });
  }
};
const TIP = { backgroundColor: 'rgba(21,21,41,.96)', titleColor: PAL.bone, bodyColor: '#b1afc8', borderColor: 'rgba(188,186,255,.24)', borderWidth: 1, padding: 10, cornerRadius: 12, boxPadding: 4, titleFont: { family: 'Instrument Sans, system-ui, sans-serif', weight: '600' }, bodyFont: { family: 'JetBrains Mono, monospace', size: 12 } };
const TICK = { color: PAL.mute, font: { size: 11, family: 'JetBrains Mono, monospace' } };
function lineOptions(o = {}) {
  return {
    responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, animation: { duration: 900, easing: 'easeOutQuart' },
    plugins: {
      legend: o.legend ? { display: true, position: 'bottom', labels: { color: '#b1afc8', usePointStyle: true, pointStyle: 'circle', boxWidth: 7, boxHeight: 7, padding: 18, font: { size: 12, family: 'Instrument Sans, system-ui, sans-serif' } } } : { display: false },
      tooltip: TIP, thresholds: o.thresholds ? { lines: o.thresholds } : false
    },
    scales: {
      x: { grid: { display: false }, border: { color: PAL.grid }, ticks: { ...TICK, autoSkip: false, maxRotation: 0, callback(v, i, all) { return (i % 6 === 0 || i === all.length - 1) ? this.getLabelForValue(v) : ''; } } },
      y: { grid: { color: PAL.grid }, border: { display: false }, ticks: TICK, min: o.min, max: o.max }
    },
    elements: { point: { radius: 0, hoverRadius: 4 }, line: { tension: .3 } }
  };
}
function makeChart(id, labels, datasets, o = {}) {
  destroyChart(id);
  const ctx = $(id); if (!ctx) return;
  const ds = datasets.map(d => ({
    borderWidth: 2.2, fill: !!d.area, pointRadius: 0, cubicInterpolationMode: 'monotone', ...d,
    backgroundColor: d.area ? (c => { const g = c.chart.ctx.createLinearGradient(0, 0, 0, c.chart.height); g.addColorStop(0, d.borderColor + '55'); g.addColorStop(1, d.borderColor + '00'); return g; }) : undefined
  }));
  chartInstances[id] = new Chart(ctx, { type: 'line', data: { labels, datasets: ds }, options: lineOptions(o), plugins: [thresholdPlugin] });
}

/* ─── SPARKLINES ─── */
function drawSparkline(canvas, data, color) {
  if (!canvas || !data || !data.length) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1), cw = canvas.clientWidth || 120, ch = canvas.clientHeight || 28;
  if (canvas.width !== Math.round(cw * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cw, ch);
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const pts = data.map((v, i) => [(i / (data.length - 1)) * (cw - 6) + 3, ch - 3 - ((v - min) / range) * (ch - 8)]);
  const g = ctx.createLinearGradient(0, 0, 0, ch); g.addColorStop(0, color + '44'); g.addColorStop(1, color + '00');
  ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.lineTo(pts[pts.length - 1][0], ch); ctx.lineTo(pts[0][0], ch); ctx.closePath(); ctx.fillStyle = g; ctx.fill();
  ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.stroke();
  const [lx, ly] = pts[pts.length - 1]; ctx.beginPath(); ctx.arc(lx, ly, 2.4, 0, TAU); ctx.fillStyle = color; ctx.fill();
}
function drawAllSparklines(container) {
  requestAnimationFrame(() => container.querySelectorAll('canvas[data-idx]').forEach(c => {
    const p = patients[+c.dataset.idx]; drawSparkline(c, p.lstmS, riskHex(p.risk));
  }));
}

/* ─── LIVE ECG (simulated from the patient's heart rate) ─── */
const Monitors = (() => {
  const traces = new Set(); let raf = 0, last = 0;
  const g = (x, s) => Math.exp(-0.5 * (x / s) * (x / s));
  function makeRhythm(hr, irregular) {
    const rng = mulberry32(Math.round(hr * 13) + (irregular ? 7 : 0)), r = { beats: [], next: 0.35 };
    r.extend = t => {
      while (r.next < t + 1.5) {
        const rr = 60 / hr;
        if (irregular && rng() < 0.12) { r.beats.push({ t: r.next - rr * 0.3, type: 'pvc' }); r.next += rr * 1.02; continue; }
        r.beats.push({ t: r.next, type: 'n' }); r.next += rr * (1 + (rng() - 0.5) * (irregular ? 0.28 : 0.025));
      }
      if (r.beats.length > 80) r.beats = r.beats.filter(b => b.t > t - 3);
    };
    return r;
  }
  function beatEcg(dt, type, st) {
    if (type === 'pvc') return 1.05 * g(dt, 0.03) - 0.5 * g(dt - 0.2, 0.07) - 0.1 * g(dt + 0.03, 0.02);
    let v = 0.12 * g(dt + 0.17, 0.026) - 0.1 * g(dt + 0.035, 0.009) + g(dt, 0.0105) - 0.22 * g(dt - 0.032, 0.011);
    v += st ? (0.09 * g(dt - 0.3, 0.05) - 0.09 * g(dt - 0.14, 0.06)) : 0.28 * g(dt - 0.27, 0.048);
    return v;
  }
  function sample(tr, t) {
    let v = 0;
    if (tr.kind === 'ecg') {
      for (const b of tr.rhythm.beats) { const dt = t - b.t; if (dt > -0.4 && dt < 0.7) v += beatEcg(dt, b.type, tr.st); }
      v += 0.018 * Math.sin(t * 1.3) + 0.01 * Math.sin(t * 61.7) * Math.sin(t * 13.3);
      return tr.baseY - v * tr.scale;
    }
    for (const b of tr.rhythm.beats) { const x = t - b.t - 0.1; if (x > -0.2 && x < 0.9) v += g(x - 0.14, 0.07) + 0.34 * g(x - 0.4, 0.09); }
    return tr.baseY - v * tr.scale * tr.amp;
  }
  function size(tr) {
    const c = tr.canvas, cw = Math.floor(c.clientWidth), ch = Math.floor(c.clientHeight);
    if (!cw || !ch) return false;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (c.width !== cw * dpr || c.height !== ch * dpr) { c.width = cw * dpr; c.height = ch * dpr; tr.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); tr.ys = new Float32Array(cw).fill(NaN); tr.w = cw; tr.h = ch; }
    tr.baseY = tr.kind === 'ecg' ? ch * 0.68 : ch * 0.9; tr.scale = tr.kind === 'ecg' ? ch * 0.54 : ch * 0.7;
    return true;
  }
  function step(tr, dt) {
    const prev = tr.px; tr.px += tr.speed * dt;
    for (let p = Math.floor(prev) + 1; p <= Math.floor(tr.px); p++) { const t = p / tr.speed; tr.rhythm.extend(t); tr.ys[p % tr.w] = sample(tr, t); }
    tr.x = tr.px % tr.w;
  }
  function draw(tr, sweep) {
    const { ctx, w, h, ys } = tr; ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1.9; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.strokeStyle = tr.color; ctx.shadowColor = tr.color; ctx.shadowBlur = 8;
    const gap = sweep ? Math.max(14, w * 0.03) : 0, cx = tr.x;
    ctx.beginPath(); let pen = false;
    for (let i = 0; i < w; i++) {
      const inGap = sweep && ((i > cx && i < cx + gap) || (cx + gap > w && i < cx + gap - w));
      if (inGap || Number.isNaN(ys[i])) { pen = false; continue; }
      pen ? ctx.lineTo(i, ys[i]) : ctx.moveTo(i, ys[i]); pen = true;
    }
    ctx.stroke(); ctx.shadowBlur = 0;
    if (sweep && !Number.isNaN(ys[Math.floor(cx) % w])) { ctx.beginPath(); ctx.arc(cx, ys[Math.floor(cx) % w], 3, 0, TAU); ctx.fillStyle = tr.color; ctx.shadowColor = tr.color; ctx.shadowBlur = 12; ctx.fill(); ctx.shadowBlur = 0; }
  }
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    traces.forEach(tr => { if (!tr.canvas.isConnected || !size(tr)) return; step(tr, dt); draw(tr, true); });
    raf = traces.size ? requestAnimationFrame(frame) : 0;
  }
  function addTrace(canvas, kind, o, rhythm) {
    if (!canvas) return;
    const tr = { canvas, ctx: canvas.getContext('2d'), kind, rhythm, st: !!o.st, speed: 120, px: 0, x: 0, w: 0, h: 0,
      amp: 0.45 + 0.55 * clamp(((o.spo2 ?? 98) - 70) / 28, 0, 1), color: kind === 'ecg' ? o.color : (o.color2 || o.color) };
    if (!size(tr)) return;
    if (calm) { step(tr, tr.w / tr.speed); tr.x = -100; draw(tr, false); return; }
    traces.add(tr);
  }
  return {
    mount(o) { const rhythm = makeRhythm(o.hr, o.irregular); addTrace(o.ecg, 'ecg', o, rhythm); addTrace(o.pleth, 'pleth', o, rhythm); if (traces.size && !raf) { last = performance.now(); raf = requestAnimationFrame(frame); } },
    stopAll() { traces.clear(); if (raf) cancelAnimationFrame(raf); raf = 0; }
  };
})();

/* ─── RADAR: distance from the centre is LSTM risk ─── */
const Radar = (() => {
  const cv = $('radar'), ctx = cv.getContext('2d'), tip = $('tip');
  const SECT = TAU / WARDS, GAP = 0.05;
  let W = 0, H = 0, dpr = 1, cx = 0, cy = 0, R = 0, nodes = [], hover = -1, ext = -1, step = STEPS - 1, raf = 0, last = 0, t = 0, sweep = -Math.PI / 2, top = [], age = 0;
  const rad = u => R * (0.97 - 0.84 * u); // u: 0 = outer edge, 1 = centre
  function build() {
    nodes = patients.map((p, i) => ({ i, p, s: 0, u: 0, tu: 0, target: scoreAt(p, step), x: 0, y: 0, flash: 0,
      a: -Math.PI / 2 + (p.ward - 1) * SECT + GAP + ((p.bed - 0.5) / BEDS_MAX) * (SECT - 2 * GAP) }));
    retarget();
  }
  function retarget() {
    nodes.forEach(n => { n.target = scoreAt(n.p, step); });
    // Position: linear up to the 0.65 ring; inside the high-risk core the model is so confident that scores
    // barely differ, so distance shows rank (highest score closest to the centre).
    const highs = nodes.filter(n => n.target > 0.65).sort((a, b) => b.target - a.target || a.i - b.i);
    highs.forEach((n, k) => { n.tu = 0.45 + 0.55 * (1 - (k + 0.5) / highs.length); });
    nodes.forEach(n => { if (n.target <= 0.65) n.tu = n.target <= 0.35 ? n.target / 0.35 * 0.30 : 0.30 + (n.target - 0.35) / 0.30 * 0.15; });
    top = [...nodes].sort((a, b) => b.target - a.target).slice(0, 8).map(n => n.i);
  }
  function resize() {
    const r = cv.getBoundingClientRect(); if (!r.width || !r.height) return;
    dpr = Math.min(2, window.devicePixelRatio || 1); W = r.width; H = r.height;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cx = W / 2; cy = H / 2 + 6; R = Math.min(W, H) / 2 - 38;
  }
  function ring(s, dash, a) { ctx.beginPath(); ctx.setLineDash(dash); ctx.strokeStyle = `rgba(188,186,255,${a})`; ctx.arc(cx, cy, rad(s), 0, TAU); ctx.stroke(); ctx.setLineDash([]); }
  function draw(dt) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 1;
    // sector spokes + ward names
    for (let w = 0; w < WARDS; w++) {
      const a = -Math.PI / 2 + w * SECT, am = a + SECT / 2;
      ctx.strokeStyle = 'rgba(188,186,255,.09)'; ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * rad(1) * 0.9, cy + Math.sin(a) * rad(1) * 0.9); ctx.lineTo(cx + Math.cos(a) * (R + 8), cy + Math.sin(a) * (R + 8)); ctx.stroke();
      ctx.fillStyle = '#8381a1'; ctx.font = '500 12px "Instrument Sans", system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`Ward ${w + 1}`, cx + Math.cos(am) * (R + 24), cy + Math.sin(am) * (R + 24));
    }
    ring(0, [], 0.2); ring(0.30, [3, 5], 0.26); ring(0.45, [3, 5], 0.3); ring(1, [], 0.12);
    ctx.fillStyle = '#8381a1'; ctx.font = '500 10.5px "JetBrains Mono", monospace'; ctx.textAlign = 'left';
    ctx.fillText('0.35', cx + 5, cy - rad(0.30) + 11); ctx.fillText('0.65', cx + 5, cy - rad(0.45) + 11); ctx.fillText('critical', cx - 18, cy + 4);
    // heat haze at the core
    const hz = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad(0.45)); hz.addColorStop(0, 'rgba(255,68,56,.16)'); hz.addColorStop(1, 'rgba(255,68,56,0)');
    ctx.fillStyle = hz; ctx.beginPath(); ctx.arc(cx, cy, rad(0.45), 0, TAU); ctx.fill();
    // sweep
    if (!calm) sweep = (sweep + dt * 0.5) % TAU;
    if (!calm && ctx.createConicGradient) {
      const cg = ctx.createConicGradient(sweep - 1.2, cx, cy);
      cg.addColorStop(0, 'rgba(242,239,252,0)'); cg.addColorStop(0.19, 'rgba(242,239,252,.13)'); cg.addColorStop(0.1905, 'rgba(242,239,252,0)'); cg.addColorStop(1, 'rgba(242,239,252,0)');
      ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.99, 0, TAU); ctx.fill();
      ctx.strokeStyle = 'rgba(242,239,252,.45)'; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(sweep) * R * 0.99, cy + Math.sin(sweep) * R * 0.99); ctx.stroke();
    }
    // move nodes
    const k = calm ? 1 : 1 - Math.exp(-dt * (age < 4 ? 1.5 : 3.6));
    nodes.forEach(n => {
      n.s += (n.target - n.s) * k; n.u += (n.tu - n.u) * k;
      const r = rad(n.u); n.x = cx + r * Math.cos(n.a); n.y = cy + r * Math.sin(n.a);
      const diff = ((sweep - n.a) % TAU + TAU) % TAU; n.flash = (!calm && diff < 0.7) ? 1 - diff / 0.7 : 0;
    });
    // ripples on the sickest patients
    if (!calm) top.forEach((i, j) => {
      const n = nodes[i]; if (n.s < 0.6) return; const ph = (t * 0.55 + j * 0.31) % 1;
      ctx.strokeStyle = heatCss(Math.min(n.s, 0.93), (1 - ph) * 0.5); ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(n.x, n.y, 4 + ph * 30, 0, TAU); ctx.stroke();
    });
    // halos (additive), then cores
    ctx.globalCompositeOperation = 'lighter';
    nodes.forEach(n => {
      const hot = n.s > 0.35 || n.flash > 0.01; if (!hot) return;
      const c = heatRGB(Math.min(n.s, 0.93)), a = (n.s > 0.65 ? 0.17 : 0.07) + n.flash * 0.28;
      ctx.fillStyle = rgba(c, a); ctx.beginPath(); ctx.arc(n.x, n.y, 7 + n.s * 6 + n.flash * 5, 0, TAU); ctx.fill();
    });
    ctx.globalCompositeOperation = 'source-over';
    nodes.forEach(n => {
      const c = heatRGB(Math.min(n.s, 0.93)), beat = calm ? 0 : Math.max(0, Math.sin(t * (n.p.hr / 60) * TAU));
      const rr = (2.5 + n.s * 2.3) * (1 + beat * (n.s > 0.5 ? 0.45 : 0.18) + n.flash * 0.35);
      ctx.fillStyle = rgba(c, 0.78 + n.flash * 0.22); ctx.beginPath(); ctx.arc(n.x, n.y, rr, 0, TAU); ctx.fill();
    });
    // hover / linked highlight
    const hi = hover >= 0 ? hover : ext;
    if (hi >= 0) {
      const n = nodes[hi];
      ctx.strokeStyle = '#f2effc'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(n.x, n.y, 10 + (calm ? 0 : Math.sin(t * 5) * 1.5), 0, TAU); ctx.stroke();
      ctx.strokeStyle = 'rgba(242,239,252,.28)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(n.x, n.y); ctx.stroke();
    }
  }
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now; t += dt; age += dt; draw(dt);
    raf = requestAnimationFrame(frame);
  }
  function hit(x, y) { let best = -1, bd = 13; nodes.forEach(n => { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.i; } }); return best; }
  function showTip(i, ev) {
    if (i < 0) { tip.classList.remove('on'); return; }
    const p = patients[i], s = nodes[i].target;
    tip.innerHTML = `<div class="id">${p.id} · ${p.age}y ${p.sex}</div><span class="num" style="color:${heatCss(s)}">${fmtS(s)}</span><small>Ward ${p.ward} · Bed ${p.bed}</small><small>HR ${p.hr} · SpO<sub>2</sub> ${p.spo2.toFixed(1)}%</small>`;
    let x = ev.clientX + 18, y = ev.clientY + 14; const w = tip.offsetWidth || 180, h = tip.offsetHeight || 120;
    if (x + w > innerWidth - 10) x = ev.clientX - w - 18; if (y + h > innerHeight - 10) y = ev.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px'; tip.classList.add('on');
  }
  const local = ev => { const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  cv.addEventListener('pointermove', ev => { const [x, y] = local(ev); hover = hit(x, y); cv.classList.toggle('hit', hover >= 0); showTip(hover, ev); });
  cv.addEventListener('pointerleave', () => { hover = -1; showTip(-1); cv.classList.remove('hit'); });
  cv.addEventListener('click', ev => { const [x, y] = local(ev), i = hit(x, y); if (i >= 0) { showTip(-1); openDossier(i, ev.clientX, ev.clientY); } });
  new ResizeObserver(resize).observe($('stage'));
  build();
  return {
    start() { resize(); if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); } },
    stop() { if (raf) cancelAnimationFrame(raf); raf = 0; },
    restartIntro() { age = 0; nodes.forEach(n => { n.s = 0; n.u = 0; }); },
    setStep(s) { step = s; retarget(); }, retarget, setExternal(i) { ext = i; },
    pos(i) { return [nodes[i].x, nodes[i].y]; }, rect: () => cv.getBoundingClientRect()
  };
})();

/* ─── REPLAY SCRUBBER + WARD SIDE ─── */
let viewStep = STEPS - 1, playing = false, playTimer = null;
function meanAt(step) { return patients.reduce((s, p) => s + scoreAt(p, step), 0) / patients.length; }
function drawScrub() {
  const cvs = $('scrub-spark'), dprr = Math.min(2, window.devicePixelRatio || 1), w = cvs.clientWidth, h = cvs.clientHeight; if (!w) return;
  cvs.width = w * dprr; cvs.height = h * dprr; const c = cvs.getContext('2d'); c.setTransform(dprr, 0, 0, dprr, 0, 0); c.clearRect(0, 0, w, h);
  const vals = Array.from({ length: STEPS }, (_, i) => meanAt(i)), mn = Math.min(...vals) - 0.01, mx = Math.max(...vals) + 0.01;
  const pt = i => [i / (STEPS - 1) * w, h - 12 - (vals[i] - mn) / (mx - mn) * (h - 26)];
  const g = c.createLinearGradient(0, 0, w, 0); g.addColorStop(0, 'rgba(90,162,255,.5)'); g.addColorStop(1, 'rgba(255,68,56,.55)');
  const fill = c.createLinearGradient(0, 0, 0, h); fill.addColorStop(0, 'rgba(242,239,252,.14)'); fill.addColorStop(1, 'rgba(242,239,252,0)');
  c.beginPath(); vals.forEach((_, i) => { const [x, y] = pt(i); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.lineTo(w, h); c.lineTo(0, h); c.closePath(); c.fillStyle = fill; c.fill();
  c.beginPath(); vals.forEach((_, i) => { const [x, y] = pt(i); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.strokeStyle = g; c.lineWidth = 2; c.stroke();
  c.fillStyle = 'rgba(188,186,255,.35)'; for (let i = 0; i < STEPS; i++) c.fillRect(i / (STEPS - 1) * w - 0.5, h - 6, 1, 6);
}
function updateScrubUI() {
  $('scrub-thumb').style.left = (viewStep / (STEPS - 1) * 100) + '%';
  $('scrub-label').textContent = viewStep === STEPS - 1 ? 'Now' : `−${STEPS - 1 - viewStep}`;
  $('scrub-track').setAttribute('aria-valuenow', viewStep);
  $('scrub-track').setAttribute('aria-valuetext', viewStep === STEPS - 1 ? 'Now' : `${STEPS - 1 - viewStep} readings ago`);
}
function setStep(s) {
  s = clamp(Math.round(s), 0, STEPS - 1); if (s === viewStep) return;
  viewStep = s; Radar.setStep(s); updateScrubUI(); updateWardSide();
}
function stopPlay() { playing = false; clearInterval(playTimer); $('play-icon').firstElementChild.setAttribute('href', '#i-play'); }
function togglePlay() {
  if (playing) { stopPlay(); return; }
  if (viewStep >= STEPS - 1) setStep(0);
  playing = true; $('play-icon').firstElementChild.setAttribute('href', '#i-pause');
  playTimer = setInterval(() => { if (viewStep >= STEPS - 1) { stopPlay(); return; } setStep(viewStep + 1); }, 560);
}
(function initScrub() {
  const tr = $('scrub-track'); let drag = false;
  const at = ev => { const r = tr.getBoundingClientRect(); setStep(clamp((ev.clientX - r.left) / r.width, 0, 1) * (STEPS - 1)); };
  tr.addEventListener('pointerdown', ev => { drag = true; tr.setPointerCapture(ev.pointerId); stopPlay(); at(ev); });
  tr.addEventListener('pointermove', ev => { if (drag) at(ev); });
  tr.addEventListener('pointerup', () => { drag = false; });
  tr.addEventListener('keydown', ev => {
    const m = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[ev.key];
    if (m) { ev.preventDefault(); stopPlay(); setStep(viewStep + m); }
    if (ev.key === 'Home') { stopPlay(); setStep(0); } if (ev.key === 'End') { stopPlay(); setStep(STEPS - 1); }
  });
  new ResizeObserver(drawScrub).observe(tr);
})();

const countState = {};
function countTo(id, v) {
  const el = $(id), from = countState[id] ?? 0; countState[id] = v;
  if (calm || from === v) { el.textContent = v; return; }
  const t0 = performance.now(), d = 650;
  (function tick(now) { const k = clamp((now - t0) / d, 0, 1), e = 1 - Math.pow(1 - k, 3); el.textContent = Math.round(lerp(from, v, e)); if (k < 1 && countState[id] === v) requestAnimationFrame(tick); })(t0);
}
function updateWardSide() {
  let hi = 0, md = 0, lo = 0;
  patients.forEach(p => { const s = scoreAt(p, viewStep); s > 0.65 ? hi++ : s > 0.35 ? md++ : lo++; });
  countTo('c-high', hi); countTo('c-med', md); countTo('c-low', lo); $('c-total').textContent = patients.length;
  const list = patients.map((p, i) => [i, scoreAt(p, viewStep)]).sort((a, b) => b[1] - a[1]).slice(0, 8);
  $('queue-list').innerHTML = list.map(([i, s], n) => {
    const p = patients[i];
    return `<button class="q-item" onclick="openDossier(${i},event.clientX,event.clientY)" onmouseenter="Radar.setExternal(${i})" onmouseleave="Radar.setExternal(-1)" onfocus="Radar.setExternal(${i})" onblur="Radar.setExternal(-1)">
      <span class="rk">${n + 1}</span><span><span class="id">${p.id}</span><span class="meta">${p.age}y ${p.sex} · Ward ${p.ward}</span></span><span class="num" style="color:${heatCss(s)}">${fmtS(s)}</span>
      <span class="q-bar"><i style="width:${(s * 100).toFixed(0)}%;background:${heatCss(s)}"></i></span></button>`;
  }).join('');
  const [ti, ts] = list[0], chip = $('alarm-chip');
  if (ts > 0.65) {
    chip.hidden = false; chip.onclick = ev => openDossier(ti, ev.clientX, ev.clientY);
    chip.innerHTML = `<i></i><span><b>${ts > 0.7 ? 'Critical' : 'High risk'} · ${patients[ti].id}</b> <span>LSTM ${fmtS(ts)} · SpO<sub>2</sub> ${patients[ti].spo2.toFixed(1)}%</span></span>`;
  } else chip.hidden = true;
}
function updateCounts() {
  const n = patients.length, c = r => patients.filter(p => p.risk === r).length;
  const crit = patients.filter(p => p.lstm > 0.7).length, warn = patients.filter(p => p.lstm > 0.35 && p.lstm <= 0.7).length;
  $('fc-all').textContent = `(${n})`; $('fc-high').textContent = `(${c('HIGH')})`; $('fc-med').textContent = `(${c('MEDIUM')})`; $('fc-low').textContent = `(${c('LOW')})`;
  $('ac-all').textContent = `(${crit + warn})`; $('ac-crit').textContent = `(${crit})`; $('ac-warn').textContent = `(${warn})`;
  $('patients-sub').textContent = `${n} active ICU admissions across ${WARDS} wards. Select a patient to open their dossier.`;
  $('insights-sub').textContent = `Population statistics across ${n} patients.`;
}

/* ─── SMALL UI PIECES ─── */
const ico = n => `<svg class="i"><use href="#i-${n}"/></svg>`;
function badge(risk) { const c = risk === 'HIGH' ? 'b-high' : risk === 'MEDIUM' ? 'b-med' : 'b-low'; return `<span class="badge ${c}"><i></i>${riskLabel(risk)}</span>`; }
const openAttr = i => `onclick="openDossier(${i},event.clientX,event.clientY)" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();var r=this.getBoundingClientRect();openDossier(${i},r.left+r.width/2,r.top+r.height/2)}" tabindex="0"`;
const flag = (v, crit, warn) => v > crit ? 'f-hot' : v > warn ? 'f-amber' : '';
const labels24 = () => Array.from({ length: STEPS }, (_, i) => i === STEPS - 1 ? 'Now' : `-${STEPS - 1 - i}`);
function sortList(list, sort) {
  const f = { 'lstm-desc': (a, b) => b.lstm - a.lstm, 'lstm-asc': (a, b) => a.lstm - b.lstm, 'age-desc': (a, b) => b.age - a.age, 'hr-desc': (a, b) => b.hr - a.hr, 'id': (a, b) => a.id.localeCompare(b.id) }[sort];
  return f ? [...list].sort(f) : list;
}

/* ─── PATIENTS ─── */
let currentFilter = 'all', patientsView = 'list';
function setPatientsView(v, el) {
  patientsView = v; document.querySelectorAll('#view-list-btn,#view-beds-btn').forEach(b => b.classList.remove('active')); el.classList.add('active');
  $('patients-list-wrap').hidden = v !== 'list'; $('patients-beds').hidden = v !== 'beds'; filterPatients();
}
function setFilter(f, el) { currentFilter = f; document.querySelectorAll('#patient-filters .chip').forEach(b => b.classList.remove('active')); el.classList.add('active'); filterPatients(); }
function filterPatients() {
  const q = ($('patient-search').value || '').toLowerCase();
  let list = patients.filter(p => {
    const mr = currentFilter === 'all' || (currentFilter === 'high' && p.risk === 'HIGH') || (currentFilter === 'med' && p.risk === 'MEDIUM') || (currentFilter === 'low' && p.risk === 'LOW');
    return mr && (!q || p.id.toLowerCase().includes(q) || String(p.age).includes(q));
  });
  list = sortList(list, $('patient-sort').value);
  if (patientsView === 'list') {
    const box = $('patients-list');
    box.innerHTML = list.length ? list.map(p => {
      const i = patients.indexOf(p), c = riskCls(p.risk);
      return `<div class="prow r-${c}" ${openAttr(i)} aria-label="${p.id}, ${riskLabel(p.risk)}">
        <span class="id">${p.id}</span><span>${p.age}y ${p.sex}</span><span class="mono">W${p.ward} · B${p.bed}</span>
        <span class="mono ${flag(p.hr, 130, 100)}">${p.hr}</span>
        <span class="mono ${p.spo2 < 88 ? 'f-hot' : p.spo2 < 94 ? 'f-amber' : ''}">${p.spo2.toFixed(1)}</span>
        <span class="mono">${p.sbp}/${p.dbp}</span>
        <span class="mono ${flag(p.temp, 39, 38)}">${p.temp.toFixed(1)}</span>
        <canvas data-idx="${i}" aria-hidden="true"></canvas>
        <span class="hbar"><i style="width:${(p.lstm * 100).toFixed(0)}%;background:${heatCss(p.lstm)}"></i></span>
        <span class="sc"><span class="num t-${c}">${fmtS(p.lstm)}</span></span></div>`;
    }).join('') : '<div class="empty">No patients match this search.</div>';
    drawAllSparklines(box);
  } else {
    const box = $('patients-beds'); let html = '';
    for (let w = 1; w <= WARDS; w++) {
      const inW = list.filter(p => p.ward === w).sort((a, b) => a.bed - b.bed); if (!inW.length) continue;
      html += `<div><div class="ward-name">Ward ${w}<small>${inW.length} patients · ${inW.filter(p => p.risk === 'HIGH').length} high risk</small></div><div class="beds">${inW.map(p =>
        `<button class="bed k-${riskCls(p.risk)}" style="--k:${(0.25 + p.lstm * 0.75).toFixed(2)}" title="${p.id} · LSTM ${fmtS(p.lstm)}" aria-label="${p.id}, LSTM ${fmtS(p.lstm)}" onclick="openDossier(${patients.indexOf(p)},event.clientX,event.clientY)">${p.bed}</button>`).join('')}</div></div>`;
    }
    box.innerHTML = html || '<div class="empty">No patients match this search.</div>';
  }
}

/* ─── ALERTS ─── */
let alertFilter = 'all';
function setAlertFilter(f, el) { alertFilter = f; document.querySelectorAll('#page-alerts .chip').forEach(b => b.classList.remove('active')); el.classList.add('active'); renderAlerts(); }
function renderAlerts() {
  let list = patients.filter(p => p.lstm > 0.35);
  if (alertFilter === 'critical') list = list.filter(p => p.lstm > 0.7);
  if (alertFilter === 'warning') list = list.filter(p => p.lstm <= 0.7);
  const groups = [['Above the threshold for under 3 hours', 0, 3], ['3 to 12 hours', 3, 12], ['More than 12 hours', 12, 99]];
  let html = '';
  groups.forEach(([title, lo, hi]) => {
    const g = list.filter(p => p.alertH > lo && p.alertH <= hi).sort((a, b) => a.alertH - b.alertH || b.lstm - a.lstm); if (!g.length) return;
    html += `<div class="tl-mark">${title} <span style="font-weight:400;color:var(--text-3)">· ${g.length}</span></div>` + g.map(p => {
      const crit = p.lstm > 0.7;
      return `<div class="al ${crit ? 'critical' : 'warning'}" ${openAttr(patients.indexOf(p))}>
        <div><div class="t">${p.id}<span>${crit ? 'Critical' : 'Warning'} · ${p.age}y ${p.sex} · Ward ${p.ward}</span></div><div class="v">SpO2 ${p.spo2.toFixed(1)}% · HR ${p.hr} bpm · Temp ${p.temp.toFixed(1)}°C · RR ${p.rr}</div></div>
        <canvas data-idx="${patients.indexOf(p)}" aria-hidden="true"></canvas>
        <div class="sc"><span class="num">${fmtS(p.lstm)}</span><small>for ${p.alertH >= STEPS ? STEPS + 'h+' : p.alertH + 'h'}</small></div></div>`;
    }).join('');
  });
  const box = $('alerts-list'); box.innerHTML = html || '<div class="empty">No alerts in this category.</div>'; drawAllSparklines(box);
}

/* ─── INSIGHTS ─── */
const avg = (arr, f) => arr.reduce((s, p) => s + f(p), 0) / arr.length;
function renderInsights() {
  renderPerformance();
  countTo('kpi-hr', Math.round(avg(patients, p => p.hr)));
  $('kpi-spo2').textContent = avg(patients, p => p.spo2).toFixed(1); $('kpi-lstm').textContent = avg(patients, p => p.lstm).toFixed(2); $('kpi-temp').textContent = avg(patients, p => p.temp).toFixed(1);
  requestAnimationFrame(() => {
    ['chart-risk-dist', 'chart-pie', 'chart-wards', 'chart-age', 'chart-gender'].forEach(destroyChart);
    const axis = { grid: { color: PAL.grid }, border: { display: false }, ticks: TICK };
    const bar = { responsive: true, maintainAspectRatio: false, animation: { duration: 1000, easing: 'easeOutQuart' }, plugins: { legend: { display: false }, tooltip: TIP }, scales: { x: { ...axis, grid: { display: false } }, y: axis } };
    const bins = new Array(10).fill(0); patients.forEach(p => { bins[Math.min(9, Math.floor(p.lstm * 10))]++; });
    chartInstances['chart-risk-dist'] = new Chart($('chart-risk-dist'), { type: 'bar', data: { labels: ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], datasets: [{ data: bins, backgroundColor: bins.map((_, i) => heatCss(i / 10 + 0.05)), borderRadius: 6, borderSkipped: false }] }, options: bar });
    const cnt = r => patients.filter(p => p.risk === r).length;
    const donut = { responsive: true, maintainAspectRatio: false, cutout: '70%', animation: { duration: 1000 }, plugins: { legend: { display: true, position: 'bottom', labels: { color: '#b1afc8', usePointStyle: true, pointStyle: 'circle', boxWidth: 7, boxHeight: 7, padding: 16, font: { size: 12, family: 'Instrument Sans, system-ui, sans-serif' } } }, tooltip: TIP } };
    chartInstances['chart-pie'] = new Chart($('chart-pie'), { type: 'doughnut', data: { labels: ['High risk', 'Medium', 'Stable'], datasets: [{ data: [cnt('HIGH'), cnt('MEDIUM'), cnt('LOW')], backgroundColor: [PAL.hot, PAL.amber, PAL.cold], borderColor: '#07070f', borderWidth: 4 }] }, options: donut });
    const wardAvg = Array.from({ length: WARDS }, (_, w) => { const g = patients.filter(p => p.ward === w + 1); return +(avg(g, p => p.lstm)).toFixed(3); });
    chartInstances['chart-wards'] = new Chart($('chart-wards'), { type: 'polarArea', data: { labels: wardAvg.map((_, i) => `Ward ${i + 1}`), datasets: [{ data: wardAvg, backgroundColor: wardAvg.map(v => heatCss(v + 0.08, 0.6)), borderColor: wardAvg.map(v => heatCss(v + 0.08)), borderWidth: 1.5 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: 1200 }, plugins: { legend: { display: false }, tooltip: TIP }, scales: { r: { grid: { color: PAL.grid }, angleLines: { color: PAL.grid }, ticks: { display: false }, min: 0, max: 0.6, pointLabels: { display: true, color: '#8381a1', font: { size: 11 } } } } } });
    const ageGroups = ['18-30', '31-45', '46-60', '61-75', '75+'];
    const ageRisk = ageGroups.map(g => { const [a, b] = g.includes('+') ? [75, 200] : [parseInt(g), parseInt(g.split('-')[1])]; const grp = patients.filter(p => p.age >= a && p.age <= b); return grp.length ? +(avg(grp, p => p.lstm)).toFixed(2) : 0; });
    chartInstances['chart-age'] = new Chart($('chart-age'), { type: 'bar', data: { labels: ageGroups, datasets: [{ data: ageRisk, backgroundColor: ageRisk.map(v => heatCss(v + 0.05)), borderRadius: 6, borderSkipped: false }] }, options: { ...bar, scales: { ...bar.scales, y: { ...axis, min: 0, max: 1 } } } });
    const male = patients.filter(p => p.sex === 'M').length;
    chartInstances['chart-gender'] = new Chart($('chart-gender'), { type: 'doughnut', data: { labels: ['Male', 'Female'], datasets: [{ data: [male, patients.length - male], backgroundColor: [PAL.bone, PAL.mute], borderColor: '#07070f', borderWidth: 4 }] }, options: donut });
  });
}


function renderPerformance() {
  const M = DATA.metrics, pct = v => Math.round(v * 100) + '%';
  $('perf-sub').textContent = `Measured on ${M.testPatients} held-out patients (${M.testSequences.toLocaleString()} sequences) that the model never saw in training. Threshold ${M.threshold}.`;
  $('perf-auc').textContent = M.auc.toFixed(3); $('perf-acc').textContent = pct(M.accuracy); $('perf-rec').textContent = pct(M.recall[1]); $('perf-prec').textContent = pct(M.precision[1]);
  const [[tn, fp], [fn, tp]] = M.cm, cell = (n, good, lbl) => `<div class="cm-c ${good ? 'ok' : 'bad'}"><b>${n.toLocaleString()}</b><small>${lbl}</small></div>`;
  $('perf-cm').innerHTML = `<span></span><span class="cm-h">Predicted low</span><span class="cm-h">Predicted high</span><span class="cm-h r">Actually low</span>${cell(tn, 1, 'correct')}${cell(fp, 0, 'false alarm')}<span class="cm-h r">Actually high</span>${cell(fn, 0, 'missed')}${cell(tp, 1, 'caught')}`;
  $('perf-base').innerHTML = `A simple baseline, ${M.baselineName.toLowerCase()}, scores <b>${M.baselineAuc.toFixed(3)}</b> on the same patients. The LSTM adds <b>${(M.auc - M.baselineAuc).toFixed(3)}</b> AUC by using the last 10 hours.`;
  requestAnimationFrame(() => {
    destroyChart('chart-roc');
    const g = $('chart-roc').getContext('2d');
    chartInstances['chart-roc'] = new Chart($('chart-roc'), {
      type: 'line',
      data: { datasets: [
        { label: `LSTM (AUC ${M.auc.toFixed(3)})`, data: M.roc.map(([x, y]) => ({ x, y })), borderColor: PAL.bone, borderWidth: 2.4, pointRadius: 0, tension: 0.15, fill: true,
          backgroundColor: c => { const gr = c.chart.ctx.createLinearGradient(0, 0, 0, c.chart.height); gr.addColorStop(0, 'rgba(242,239,252,.22)'); gr.addColorStop(1, 'rgba(242,239,252,0)'); return gr; } },
        { label: 'Chance', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: PAL.mute, borderDash: [5, 5], borderWidth: 1.2, pointRadius: 0 }
      ] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: 1100, easing: 'easeOutQuart' }, interaction: { mode: 'nearest', intersect: false },
        plugins: { legend: { display: true, position: 'bottom', labels: { color: '#b1afc8', usePointStyle: true, pointStyle: 'circle', boxWidth: 7, boxHeight: 7, padding: 16, font: { size: 12, family: 'Instrument Sans, system-ui, sans-serif' } } }, tooltip: { ...TIP, callbacks: { label: c => `${c.dataset.label}: false alarms ${(c.parsed.x * 100).toFixed(0)}%, caught ${(c.parsed.y * 100).toFixed(0)}%` } } },
        scales: { x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate', color: PAL.mute, font: { size: 11 } }, grid: { color: PAL.grid }, border: { display: false }, ticks: TICK },
                  y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'True positive rate', color: PAL.mute, font: { size: 11 } }, grid: { color: PAL.grid }, border: { display: false }, ticks: TICK } } }
    });
  });
}

/* ─── SETTINGS ─── */
function sw(el) { el.classList.toggle('on'); el.setAttribute('aria-checked', el.classList.contains('on')); }
function saveProfile() { showToast('Profile saved'); }
function editorOptionHTML(s) {
  const sel = selectedEditor.id === s.id;
  return `<button class="dd-opt${sel ? ' selected' : ''}" onclick="selectEditor('${s.id}')"><span class="chip-av lg" style="background:${s.color}">${s.id}</span><span><span class="who">${s.name}</span><span class="role">${s.role}</span></span>${sel ? ico('check') : ''}</button>`;
}
function renderEditorOptions(filter = '') {
  const f = filter.toLowerCase(), list = STAFF.filter(s => !f || s.name.toLowerCase().includes(f) || s.role.toLowerCase().includes(f));
  const docs = list.filter(s => !s.role.includes('Nurse')), nurses = list.filter(s => s.role.includes('Nurse'));
  let html = ''; if (docs.length) html += `<div class="dd-label">Doctors</div>${docs.map(editorOptionHTML).join('')}`;
  if (nurses.length) html += `${docs.length ? '<div class="dd-div"></div>' : ''}<div class="dd-label">Nurses</div>${nurses.map(editorOptionHTML).join('')}`;
  $('editor-options-list').innerHTML = html || '<div class="dd-label">No staff found</div>';
  $('override-editor-select').innerHTML = STAFF.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
}
function filterEditorOptions() { renderEditorOptions($('editor-search-input').value); }
function selectEditor(id) {
  selectedEditor = STAFF.find(s => s.id === id);
  const av = $('selected-editor-avatar'); av.textContent = selectedEditor.id; av.style.background = selectedEditor.color; $('selected-editor-name').textContent = selectedEditor.name;
  $('editor-dropdown-trigger').classList.remove('open'); $('editor-dropdown-menu').hidden = true;
  const ts = new Date().toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit' });
  editHistory.unshift({ editor: selectedEditor, ts, action: 'Record updated' }); if (editHistory.length > 5) editHistory.pop();
  $('edit-history-list').innerHTML = editHistory.map(h => `<div><span class="chip-av" style="background:${h.editor.color}">${h.editor.id}</span><div style="flex:1"><div style="font-weight:500">${h.editor.name}</div><div class="set-hint">${h.action}</div></div><span class="mono" style="color:var(--text-3)">${h.ts}</span></div>`).join('');
  $('last-edit-ts').textContent = ts + ' IST';
  $('edited-chip').hidden = false; const tav = $('topbar-editor-avatar'); tav.textContent = selectedEditor.id; tav.style.background = selectedEditor.color; $('topbar-editor-name').textContent = selectedEditor.name;
  renderEditorOptions(); showToast(`Editor set to ${selectedEditor.name}`);
}
function toggleEditorDropdown() {
  const btn = $('editor-dropdown-trigger'), menu = $('editor-dropdown-menu'); menu.hidden = !menu.hidden; btn.classList.toggle('open', !menu.hidden);
  if (!menu.hidden) { renderEditorOptions(); setTimeout(() => $('editor-search-input')?.focus(), 30); }
}
function applyOverride() {
  const pid = $('override-patient-id').value.trim().toUpperCase(), staff = STAFF.find(s => s.id === $('override-editor-select').value);
  if (!pid) { showToast('Enter a patient ID first'); return; } showToast(`Editor override: ${pid} → ${staff.name}`);
}
document.addEventListener('click', e => {
  const menu = $('editor-dropdown-menu'), btn = $('editor-dropdown-trigger');
  if (menu && !menu.hidden && !menu.contains(e.target) && !btn.contains(e.target)) { menu.hidden = true; btn.classList.remove('open'); }
});

/* ─── NAVIGATION ─── */
let currentView = 'ward';
function moveDot() { const b = document.querySelector(`#nav button[data-view="${currentView}"]`); if (b) $('nav-dot').style.transform = `translateX(${b.offsetLeft + b.offsetWidth / 2}px)`; }
function showView(name) {
  currentView = name;
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  $(`page-${name}`).classList.add('active');
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  moveDot(); $(`page-${name}`).scrollTop = 0;
  if (name === 'ward') { Radar.start(); updateWardSide(); drawScrub(); } else { Radar.stop(); stopPlay(); }
  if (name === 'patients') filterPatients();
  if (name === 'alerts') renderAlerts();
  if (name === 'insights') renderInsights();
}
window.addEventListener('resize', moveDot);

/* ─── TOAST + CLOCK ─── */
let toastTimer;
function showToast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 2600); }
let clockTimer;
function startClock() {
  const tick = () => { $('live-clock').textContent = new Date().toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }); };
  tick(); clearInterval(clockTimer); clockTimer = setInterval(tick, 1000);
}

/* ─── BOOT BACKDROP (login) ─── */
const Boot = (() => {
  const cv = $('boot-field'), ctx = cv.getContext('2d'), r = mulberry32(5), dots = [];
  for (let i = 0; i < 120; i++) dots.push({ a: r() * TAU, rr: 0.22 + r() * 0.75, sp: (0.015 + r() * 0.05) * (r() < .5 ? -1 : 1), s: r() < 0.8 ? r() * 0.3 : 0.4 + r() * 0.6, sz: 1.1 + r() * 2.2 });
  let W = 0, H = 0, dpr = 1, raf = 0, last = 0, t = 0;
  function resize() { dpr = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; }
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now; t += dt;
    if (cv.width !== Math.round(innerWidth * dpr)) resize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const wide = W > 900, cx = wide ? W * 0.72 : W * 0.5, cy = wide ? H * 0.5 : H * 0.34, R = Math.min(W, H) * (wide ? 0.62 : 0.5), fade = calm ? 1 : clamp((t - 1.6) / 2.2, 0, 1);
    ctx.lineWidth = 1;
    [0.33, 0.58, 0.85, 1].forEach((k, j) => { ctx.strokeStyle = `rgba(188,186,255,${0.14 * fade})`; ctx.setLineDash(j % 2 ? [3, 6] : []); ctx.beginPath(); ctx.arc(cx, cy, R * k, 0, TAU); ctx.stroke(); });
    ctx.setLineDash([]);
    const sw = t * 0.5;
    if (!calm && ctx.createConicGradient) { const cg = ctx.createConicGradient(sw - 1.2, cx, cy); cg.addColorStop(0, 'rgba(242,239,252,0)'); cg.addColorStop(0.19, `rgba(242,239,252,${0.1 * fade})`); cg.addColorStop(0.1905, 'rgba(242,239,252,0)'); cg.addColorStop(1, 'rgba(242,239,252,0)'); ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill(); }
    dots.forEach(d => {
      const a = d.a + (calm ? 0 : t * d.sp), x = cx + Math.cos(a) * R * d.rr, y = cy + Math.sin(a) * R * d.rr;
      ctx.fillStyle = heatCss(d.s, 0.75 * fade); ctx.beginPath(); ctx.arc(x, y, d.sz * (d.s > 0.5 ? 1.4 : 1), 0, TAU); ctx.fill();
      if (d.s > 0.5) { ctx.strokeStyle = heatCss(d.s, 0.25 * fade * (1 - ((t * 0.6 + d.a) % 1))); ctx.beginPath(); ctx.arc(x, y, 3 + ((t * 0.6 + d.a) % 1) * 22, 0, TAU); ctx.stroke(); }
    });
    raf = requestAnimationFrame(frame);
  }
  return { start() { resize(); t = 0; last = performance.now(); if (!raf) raf = requestAnimationFrame(frame); }, stop() { cancelAnimationFrame(raf); raf = 0; } };
})();

/* ─── LOGIN / LOGOUT ─── */
function login(ev) {
  if (ev) ev.preventDefault();
  const r = $('login-btn').getBoundingClientRect(), app = $('app');
  app.style.setProperty('--ox', (r.left + r.width / 2) + 'px'); app.style.setProperty('--oy', (r.top + r.height / 2) + 'px');
  app.hidden = false; app.classList.add('entering');
  viewStep = STEPS - 1; updateScrubUI(); Radar.setStep(viewStep); Radar.restartIntro();
  renderEditorOptions(); updateCounts(); showView('ward'); startClock(); checkModelHealth();
  setTimeout(moveDot, 60); setTimeout(refreshModelPredictions, 1500);
  setTimeout(() => { $('page-login').hidden = true; Boot.stop(); app.classList.remove('entering'); }, 1250);
}
function logout() {
  closeDossier(); Radar.stop(); Monitors.stopAll(); stopPlay();
  $('app').hidden = true; $('page-login').hidden = false; Boot.start();
}

/* ─── PATIENT DOSSIER ─── */
let dossierIdx = -1;
const metaHTML = p => `<span>${p.age} years · ${p.sex === 'M' ? 'Male' : 'Female'} · Ward ${p.ward} · Bed ${p.bed} · First reading ${p.admitted}</span>${badge(p.risk)}<span class="badge b-test" title="${p.split === 'test' ? 'This patient was held out from training, so the score is a true prediction.' : 'The model was trained on this patient, so the score is not an independent test.'}">${p.split === 'test' ? 'Held-out test patient' : 'Training patient'}</span>`;
function trendSummary(p) {
  const n = 10, ch = [
    ['Heart rate', p.hrS, 'bpm', 15, 0], ['SpO2', p.spo2S, '%', 3, 1], ['Respiratory rate', p.rrS, 'breaths/min', 5, 0],
    ['Systolic BP', p.sbpS, 'mmHg', 15, 0], ['Temperature', p.tempS, '°C', 0.8, 1], ['Lactate', p.lacS, 'mmol/L', 1, 1]
  ].map(([name, s, u, scale, dp]) => { const a = s[s.length - n], b = lastOf(s); return { name, a, b, u, dp, score: Math.abs(b - a) / scale }; }).sort((x, y) => y.score - x.score);
  const parts = ch.filter(c => c.score >= 0.4).slice(0, 3).map(c => `${c.name} ${c.b > c.a ? 'rose' : 'fell'} from ${c.a.toFixed(c.dp)} to ${c.b.toFixed(c.dp)} ${c.u}`);
  const k = p.known.indexOf(true), r0 = p.lstmS[k], r1 = p.lstm;
  let s = parts.length ? `Over the last ${n} readings: ${parts.join('; ')}. ` : `Vitals were steady over the last ${n} readings. `;
  s += Math.abs(r1 - r0) >= 0.1 ? `The LSTM risk score ${r1 > r0 ? 'climbed' : 'dropped'} from ${fmtS(r0)} to ${fmtS(r1)} across the ${STEPS - k} readings it could score.` : `The LSTM risk score stayed near ${fmtS(r1)}.`;
  return s;
}

function renderGauge(p, animate) {
  const cx = 100, cy = 100, r = 78, an = animate ? ' anim' : '', st = animate ? '' : ' static';
  const pt = (s, rad = r) => { const a = Math.PI * (1 - s); return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)]; };
  const arc = (s0, s1) => { const [x0, y0] = pt(s0), [x1, y1] = pt(s1); return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`; };
  const s = clamp(p.lstm, 0, 1), [kx, ky] = pt(s), col = heatCss(s);
  const segs = [[0, 0.35, PAL.cold], [0.35, 0.65, PAL.amber], [0.65, 1, PAL.hot]];
  const paths = segs.map(([a, b, c]) => {
    let o = `<path d="${arc(a + 0.006, b - 0.006)}" stroke="${c}" opacity=".22" fill="none" stroke-width="11"/>`;
    if (s > a) o += `<path class="g-fill${an}" pathLength="1" d="${arc(a + 0.006, Math.min(b - 0.006, Math.max(a + 0.007, s)))}" stroke="${c}" fill="none" stroke-width="11"/>`;
    return o;
  }).join('');
  $('dz-gauge').innerHTML = `<svg viewBox="0 0 200 128" role="img" aria-label="LSTM risk score ${fmtS(p.lstm)}">${paths}
    <circle class="g-knob${st}" cx="${kx.toFixed(1)}" cy="${ky.toFixed(1)}" r="8.5" fill="#07070f" stroke="${col}" stroke-width="3"/>
    <text x="100" y="96" text-anchor="middle" class="g-num" fill="${col}">${fmtS(p.lstm)}</text>
    <text x="100" y="116" text-anchor="middle" class="g-lbl">LSTM risk score</text></svg>`;
}

function systemSeverity(p) {
  const c = clamp;
  return {
    resp: Math.max(c((96 - p.spo2) / 16, 0, 1), c((p.rr - 18) / 14, 0, 1)),
    cardio: Math.max(c((Math.abs(p.hr - 80) - 20) / 45, 0, 1), c((105 - p.sbp) / 25, 0, 1)),
    renal: Math.max(c((p.creatinine - 1.2) / 1.8, 0, 1), c((p.bun - 20) / 40, 0, 1), c((Math.abs(p.potassium - 4.2) - 0.8) / 1.5, 0, 1)),
    meta: Math.max(c((Math.abs(p.glucose - 125) - 55) / 110, 0, 1), c((Math.abs(p.sodium - 140) - 5) / 10, 0, 1)),
    blood: Math.max(c((p.lactate - 2) / 3, 0, 1), c((p.wbc - 11) / 7, 0, 1), c((150 - p.platelets) / 100, 0, 1), c((10 - p.hgb) / 3, 0, 1)),
    temp: Math.max(c((p.temp - 37.8) / 2.2, 0, 1), c((36 - p.temp) / 2, 0, 1))
  };
}
function roundedClosed(pts) {
  const n = pts.length, mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], m0 = mid(pts[n - 1], pts[0]);
  let d = `M${m0[0].toFixed(1)} ${m0[1].toFixed(1)}`;
  for (let i = 0; i < n; i++) { const p = pts[i], m = mid(p, pts[(i + 1) % n]); d += ` Q${p[0]} ${p[1]} ${m[0].toFixed(1)} ${m[1].toFixed(1)}`; }
  return d + 'Z';
}
function renderBody(p) {
  const S = systemSeverity(p), CX = 260;
  const half = [[13, 80], [16, 92], [44, 100], [84, 112], [98, 150], [104, 215], [116, 290], [124, 345], [116, 368], [104, 362], [98, 300], [86, 225], [76, 170], [68, 140], [64, 190], [58, 250], [68, 300], [70, 350], [62, 430], [60, 490], [54, 540], [62, 560], [14, 560], [16, 540], [18, 490], [14, 430], [12, 350], [0, 318]];
  const left = [...half].reverse().slice(1).map(([dx, y]) => [-dx, y]);
  const outline = roundedClosed([...half, ...left].map(([dx, y]) => [CX + dx, y]));
  const stat = k => S[k] < 0.2 ? ['Stable', PAL.cold] : S[k] < 0.55 ? ['Watch', PAL.amber] : ['Stressed', PAL.hot];
  const org = (k, o, cls, d, extra = '') => `<path class="organ ${cls}" style="--c:${sevColor(S[k])};--sev:${S[k].toFixed(2)};--o:${o};${extra}" d="${d}"/>`;
  const beat = (60 / clamp(p.hr, 40, 180)).toFixed(2) + 's', breath = (60 / clamp(p.rr, 8, 40)).toFixed(2) + 's', flow = (1.5 * 80 / clamp(p.hr, 40, 180)).toFixed(2) + 's';
  const call = (o, side, ax, ay, ly, key, title, vals) => {
    const lx = side === 'L' ? 150 : 450, sg = side === 'L' ? 1 : -1, ex = lx + sg * 22, [w, c] = stat(key), an = side === 'L' ? 'end' : 'start';
    return `<g style="--o:${o}"><path class="call-line" pathLength="1" d="M${ax + 40} ${ay} L${ex} ${ly} L${lx + sg * 6} ${ly}"/>
      <g class="call-t"><text class="call-n" x="${lx}" y="${ly - 8}" text-anchor="${an}">${title}</text><text class="call-s" x="${lx}" y="${ly + 8}" text-anchor="${an}" fill="${c}">${w}</text><text class="call-v" x="${lx}" y="${ly + 24}" text-anchor="${an}">${vals}</text></g></g>`;
  };
  const rows = [['resp', 'Respiratory', `SpO2 ${p.spo2.toFixed(1)}% · RR ${p.rr}`], ['cardio', 'Cardiovascular', `HR ${p.hr} · BP ${p.sbp}/${p.dbp}`], ['renal', 'Renal', `Creat ${p.creatinine.toFixed(1)} · BUN ${p.bun}`], ['meta', 'Metabolic', `Glucose ${p.glucose} · Na ${p.sodium}`], ['blood', 'Blood and circulation', `Lactate ${p.lactate.toFixed(1)} · WBC ${p.wbc.toFixed(1)}`], ['temp', 'Temperature', `${p.temp.toFixed(1)} °C`]];
  $('sys-list').innerHTML = rows.map(([k, n, v]) => { const [w, c] = stat(k); return `<div class="sys" style="--c:${c}"><i></i><div><b>${n}</b><span style="color:${c}">${w}</span><small>${v}</small></div></div>`; }).join('');
  $('body-svg').innerHTML = `
    <g transform="translate(40,0)">
      <path class="body-line" pathLength="1" d="${outline}"/>
      ${org('temp', 4, '', 'M236 50 a24 30 0 1 0 48 0 a24 30 0 1 0 -48 0 Z')}
      <g style="--c:${sevColor(S.blood)};--flow:${flow}">
        <path class="vessel" d="M262 92 L262 304"/><path class="vessel" d="M262 304 L244 352"/><path class="vessel" d="M262 304 L280 352"/>
        <path class="vessel" d="M262 112 C238 112 206 112 180 128"/><path class="vessel" d="M262 112 C286 112 318 112 344 128"/>
      </g>
      ${org('resp', 0, 'lung', 'M250 118 C232 110 206 128 200 162 C196 188 204 206 224 207 C240 208 250 196 250 178 Z', `--breath:${breath}`)}
      ${org('resp', 0, 'lung', 'M270 118 C288 110 314 128 320 162 C324 188 316 206 296 207 C280 208 270 196 270 178 Z', `--breath:${breath}`)}
      ${org('cardio', 1, 'heart', 'M276 196 C254 178 252 152 268 150 C274 150 277 154 278 158 C280 152 286 149 292 152 C306 158 300 180 276 196 Z', `--beat:${beat}`)}
      ${org('renal', 2, '', 'M232 220 C216 220 212 242 220 254 C226 262 238 256 236 244 C234 238 238 232 236 226 C235 222 233 220 232 220 Z')}
      ${org('renal', 2, '', 'M288 220 C304 220 308 242 300 254 C294 262 282 256 284 244 C286 238 282 232 284 226 C285 222 287 220 288 220 Z')}
      ${org('meta', 3, '', 'M232 268 C246 256 278 256 296 268 C282 280 248 280 232 268 Z')}
    </g>
    ${call(0, 'L', 205, 160, 140, 'resp', 'Respiratory', `SpO2 ${p.spo2.toFixed(1)}% · RR ${p.rr}`)}
    ${call(2, 'L', 216, 240, 236, 'renal', 'Renal', `Creat ${p.creatinine.toFixed(1)} · BUN ${p.bun}`)}
    ${call(4, 'L', 255, 330, 336, 'blood', 'Blood and circulation', `Lactate ${p.lactate.toFixed(1)} · WBC ${p.wbc.toFixed(1)}`)}
    ${call(1, 'R', 296, 172, 150, 'cardio', 'Cardiovascular', `HR ${p.hr} · BP ${p.sbp}/${p.dbp}`)}
    ${call(3, 'R', 296, 268, 274, 'meta', 'Metabolic', `Glucose ${p.glucose} · Na ${p.sodium}`)}
    ${call(5, 'R', 284, 46, 50, 'temp', 'Temperature', `${p.temp.toFixed(1)} °C`)}`;
}

function vitalCard(label, val, unit, cls, series, key) {
  const st = cls === 'abn' ? 'Abnormal' : cls === 'watch' ? 'Watch' : 'Normal';
  return `<div class="vt ${cls}"><small>${label}</small><span class="num">${val}</span><span class="u">${unit}</span><span class="st">${st}</span><canvas data-series="${key}" aria-hidden="true"></canvas></div>`;
}
let trendKey = 'hr';
function showTrend(key, el) {
  if (el) { document.querySelectorAll('#trend-tabs .chip').forEach(c => c.classList.remove('active')); el.classList.add('active'); }
  trendKey = key; const p = patients[dossierIdx]; if (!p) return;
  const sets = {
    hr: [{ label: 'Heart rate', data: p.hrS, borderColor: PAL.bone, area: true }],
    spo2: [{ label: 'SpO2', data: p.spo2S, borderColor: PAL.cold, area: true }],
    bp: [{ label: 'Systolic', data: p.sbpS, borderColor: PAL.bone }, { label: 'Diastolic', data: p.dbpS, borderColor: PAL.mute, borderDash: [5, 4] }],
    lab: [{ label: 'Lactate (mmol/L)', data: p.lacS, borderColor: PAL.amber }, { label: 'WBC (K/µL)', data: p.wbcS, borderColor: PAL.mute }],
    tr: [{ label: 'Temperature (°C)', data: p.tempS, borderColor: PAL.hot }, { label: 'Resp. rate (br/min)', data: p.rrS, borderColor: PAL.cold }]
  }[key];
  makeChart('chart-trend', labels24(), sets, { legend: sets.length > 1 });
}

function openDossier(idx, x, y) {
  if (!x && !y) { x = innerWidth / 2; y = innerHeight / 2; }
  const p = patients[idx], d = $('dossier'); dossierIdx = idx;
  stopPlay(); Radar.stop(); $('tip').classList.remove('on');
  d.style.setProperty('--ox', x + 'px'); d.style.setProperty('--oy', y + 'px'); d.style.setProperty('--heat', heatHex(p.lstm));
  $('dz-id').textContent = p.id; $('dz-meta').innerHTML = metaHTML(p); renderGauge(p, true);
  $('dz-hr').textContent = p.hr; $('dz-hr').style.color = heatCss(p.lstm);
  const V = [
    ['Heart rate', p.hr, 'bpm', p.hr > 130 || p.hr < 50 ? 'abn' : p.hr > 100 ? 'watch' : '', 'hrS'],
    ['SpO<sub>2</sub>', p.spo2.toFixed(1), '%', p.spo2 < 88 ? 'abn' : p.spo2 < 94 ? 'watch' : '', 'spo2S'],
    ['Blood pressure', `${p.sbp}/${p.dbp}`, 'mmHg', p.sbp < 90 ? 'abn' : (p.sbp > 160 || p.dbp < 60) ? 'watch' : '', 'sbpS'],
    ['Resp. rate', p.rr, 'br/min', p.rr > 25 ? 'abn' : p.rr > 20 ? 'watch' : '', 'rrS'],
    ['Temperature', p.temp.toFixed(1), '°C', p.temp > 39 ? 'abn' : p.temp > 38 ? 'watch' : '', 'tempS']
  ];
  $('dz-vitals').innerHTML = V.map(v => vitalCard(...v)).join('');
  const labs = [
    ['WBC', p.wbc.toFixed(1), 'K/µL', p.wbc > 11], ['Lactate', p.lactate.toFixed(1), 'mmol/L', p.lactate > 2], ['Creatinine', p.creatinine.toFixed(1), 'mg/dL', p.creatinine > 1.2],
    ['Platelets', p.platelets, 'K/µL', p.platelets < 150], ['Hemoglobin', p.hgb.toFixed(1), 'g/dL', p.hgb < 10], ['Glucose', p.glucose, 'mg/dL', p.glucose > 180 || p.glucose < 70],
    ['Sodium', p.sodium, 'mEq/L', p.sodium < 135 || p.sodium > 145], ['Potassium', p.potassium.toFixed(1), 'mEq/L', p.potassium < 3.5 || p.potassium > 5],
    ['BUN', p.bun, 'mg/dL', p.bun > 20], ['Resp. rate', p.rr, 'br/min', p.rr > 20]
  ];
  $('dz-labs').innerHTML = labs.map(([n, v, u, a]) => `<div class="lab${a ? ' abn' : ''}"><small>${n} <em>${u}</em></small><b>${v}</b></div>`).join('');
  $('dz-notes').textContent = trendSummary(p);
  renderBody(p);
  document.querySelectorAll('#trend-tabs .chip').forEach(c => c.classList.toggle('active', c.dataset.k === 'hr')); trendKey = 'hr';
  $('risk-sub').textContent = p.known.includes(false) ? 'LSTM score over the last 24 readings. The dashed part is before the model had 10 hours of history, so it shows the first score. Above 0.65 is high risk.' : 'LSTM score over the last 24 readings. Above 0.65 is high risk.';
  d.scrollTop = 0; d.setAttribute('aria-hidden', 'false');
  requestAnimationFrame(() => {
    d.classList.add('open');
    requestAnimationFrame(() => {
      const hc = heatHex(p.lstm);
      d.querySelectorAll('canvas[data-series]').forEach(c => drawSparkline(c, p[c.dataset.series], '#f2effc'));
      makeChart('chart-lstm', labels24(), [{ label: 'LSTM risk', data: p.lstmS, borderColor: hc, area: true, borderWidth: 2.6, segment: { borderDash: c => p.known[c.p0DataIndex] ? undefined : [4, 4] } }],
        { min: 0, max: 1, thresholds: [{ value: 0.65, color: PAL.hot, label: 'High 0.65' }, { value: 0.35, color: PAL.amber, label: 'Medium 0.35' }] });
      showTrend('hr');
      Monitors.stopAll();
      Monitors.mount({ ecg: $('mon-ecg'), pleth: $('mon-pleth'), hr: clamp(p.hr, 35, 190), spo2: p.spo2, irregular: p.risk === 'HIGH' && p.hr > 105, st: p.risk === 'HIGH', color: hc, color2: 'rgba(242,239,252,.85)' });
    });
  });
  refreshDetailPrediction(idx);
  d.querySelector('.close').focus({ preventScroll: true });
}
function closeDossier() {
  const d = $('dossier'); if (!d.classList.contains('open')) return;
  Monitors.stopAll(); d.classList.remove('open'); d.setAttribute('aria-hidden', 'true'); dossierIdx = -1;
  setTimeout(() => { if (dossierIdx < 0) ['chart-lstm', 'chart-trend'].forEach(destroyChart); }, 900);
  if (currentView === 'ward' && !$('app').hidden) Radar.start();
}
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  const m = $('editor-dropdown-menu'); if (m && !m.hidden) { m.hidden = true; $('editor-dropdown-trigger').classList.remove('open'); return; }
  closeDossier();
});

/* ─── BOOT ─── */
if (calm) { document.body.classList.add('calm'); $('sw-calm')?.classList.add('on'); $('sw-calm')?.setAttribute('aria-checked', 'true'); }
updateScrubUI(); Boot.start();
