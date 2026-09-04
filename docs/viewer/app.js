(() => {
  const CATALOGS = {
    labeled: {
      dataBase: "data",
      title: "Anomaly Label Viewer",
      sub: "라벨 있는 사업자 · metric 필터 · 줌 / 이동 / 값 탐색 · 라벨 목록",
      emptyIndex: "data/index.json 없음 — export_viewer.py 를 실행하세요.",
    },
    top100: {
      dataBase: "data-top100",
      title: "Top 100 사업자",
      sub: "top100.txt 샘플링 시계열 · metric 필터 · 줌 / 이동 / 값 탐색",
      emptyIndex:
        "data-top100/index.json 없음 — export_viewer.py --catalog top100 를 실행하세요.",
    },
  };
  let activeCatalog = "labeled";
  let dataBase = CATALOGS.labeled.dataBase;
  // Same mid-tone palette as labeling/tool.py SERIES_COLORWAY.
  const SERIES_COLORWAY = [
    "#3d7ab5",
    "#d9655a",
    "#3da86a",
    "#9b6bb8",
    "#d4a017",
    "#4a90a4",
    "#c06a5a",
    "#2e9a85",
    "#8e6ba8",
    "#d4833a",
    "#5b8fbf",
    "#c97b72",
    "#4caf77",
    "#a07cbc",
    "#b8a03a",
    "#5b9bd5",
    "#b8875a",
    "#3cb09a",
    "#9a7aab",
    "#c97a55",
  ];
  const M971_COL = "M971";
  const M971_TOD_KEY = "__m971_daily_avg__";
  const M971_TOD_LABEL = "M971 · 시각별 평균 (전기간)";
  const M971_TOD_COLOR = "#ff1a1a";
  const S_RATE_KEY = "S_RATE";
  const A_RATE_KEY = "A_RATE";
  const S_RATE_COLOR = "#00BFFF";
  const A_RATE_COLOR = "#e8590c";
  const RATE_METRICS = new Set([S_RATE_KEY, A_RATE_KEY]);
  const RATE_COLORS = { [S_RATE_KEY]: S_RATE_COLOR, [A_RATE_KEY]: A_RATE_COLOR };
  const REF_LINE_WIDTH = 2;
  const REF_LINE_DASH = "3px,2px";
  const REF_LINE = { width: REF_LINE_WIDTH, dash: REF_LINE_DASH };
  const graphEl = document.getElementById("graph");
  const selectEl = document.getElementById("plmn-select");
  const listEl = document.getElementById("label-list");
  const statusEl = document.getElementById("status");
  const metricListEl = document.getElementById("metric-filter-list");
  const hoverPanelEl = document.getElementById("hover-panel");
  const pageTitleEl = document.getElementById("page-title");
  const pageSubEl = document.getElementById("page-sub");
  const tabLabeledEl = document.getElementById("tab-labeled");
  const tabTop100El = document.getElementById("tab-top100");

  let catalog = [];
  let payload = null;
  let selectedId = null;
  /** pan | zoom | inspect */
  let interactionMode = "zoom";
  let dragmode = "zoom";
  let xRange = null;
  let yRange = null;
  let yAuto = true;
  let suppressRelayout = false;
  /** Show/hide anomaly overlays on the current chart (does not filter PLMNs). */
  let showAnomalies = true;
  /** Metric names currently drawn (and shown in the hover panel). */
  let visibleMetrics = new Set();
  let hoverIndex = null;
  /** Locked sample index for 값 탐색 (null = unlocked / follow hover). */
  let inspectIndex = null;
  let _resizeTimer = null;
  let _placeTipBound = false;
  let _placeOnTrace = false;
  const coarsePointer =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;

  function setStatus(text) {
    statusEl.textContent = text || "";
  }

  function catalogFromHash() {
    const raw = (location.hash || "").replace(/^#/, "").trim().toLowerCase();
    return raw === "top100" ? "top100" : "labeled";
  }

  function isLabeledCatalog() {
    return activeCatalog === "labeled";
  }

  function anomaliesEnabled() {
    return isLabeledCatalog() && showAnomalies;
  }

  function syncCatalogTabs() {
    const isLabeled = isLabeledCatalog();
    document.body.classList.toggle("catalog-labeled", isLabeled);
    document.body.classList.toggle("catalog-top100", !isLabeled);
    tabLabeledEl.classList.toggle("active", isLabeled);
    tabTop100El.classList.toggle("active", !isLabeled);
    tabLabeledEl.setAttribute("aria-selected", isLabeled ? "true" : "false");
    tabTop100El.setAttribute("aria-selected", isLabeled ? "false" : "true");
    const meta = CATALOGS[activeCatalog] || CATALOGS.labeled;
    if (pageTitleEl) pageTitleEl.textContent = meta.title;
    if (pageSubEl) pageSubEl.textContent = meta.sub;
    document.title = meta.title;
  }

  function resetViewState() {
    selectedId = null;
    payload = null;
    catalog = [];
    xRange = null;
    yRange = null;
    yAuto = true;
    hoverIndex = null;
    inspectIndex = null;
    visibleMetrics = new Set();
    interactionMode = "zoom";
    dragmode = "zoom";
    showAnomalies = isLabeledCatalog();
    syncYAutoButton();
    syncOverlayButtons();
    setInteractionMode("zoom", { redraw: false });
    if (listEl) listEl.innerHTML = "";
    if (selectEl) selectEl.innerHTML = "";
    if (metricListEl) metricListEl.innerHTML = "";
    if (hoverPanelEl) {
      hoverPanelEl.innerHTML = "<em>그래프에 커서를 올리세요.</em>";
    }
  }

  async function switchCatalog(next) {
    if (!CATALOGS[next] || next === activeCatalog) return;
    activeCatalog = next;
    dataBase = CATALOGS[next].dataBase;
    syncCatalogTabs();
    if (location.hash !== `#${next}`) {
      history.replaceState(null, "", `#${next}`);
    }
    resetViewState();
    await loadCatalog();
  }

  function allMetricNames() {
    return Object.keys(payload?.metrics || {});
  }

  function windowIndexBounds(data, x0ms, x1ms) {
    if (!data?.t?.length) return null;
    let lo = 0;
    let hi = data.t.length - 1;
    if (!(isFinite(x0ms) && isFinite(x1ms) && x1ms > x0ms)) {
      return [lo, hi];
    }
    const tms = (i) => parseTime(data.t[i]);
    let a = 0;
    let b = data.t.length - 1;
    while (a < b) {
      const m = (a + b) >> 1;
      if (tms(m) < x0ms) a = m + 1;
      else b = m;
    }
    lo = Math.max(0, a - 1);
    a = lo;
    b = data.t.length - 1;
    while (a < b) {
      const m = (a + b + 1) >> 1;
      if (tms(m) > x1ms) b = m - 1;
      else a = m;
    }
    hi = Math.min(data.t.length - 1, a + 1);
    return [lo, hi];
  }

  /** Metrics with a non-zero value in the on-screen window, ranked by sum (desc). */
  function metricsByViewSum() {
    const names = allMetricNames();
    if (!payload?.t?.length || !names.length) return names;
    const win = visibleXWindowMs();
    const bounds = win
      ? windowIndexBounds(payload, win[0], win[1])
      : [0, payload.t.length - 1];
    if (!bounds) return names;
    const [lo0, hi0] = bounds;
    const x0ms = win ? win[0] : -Infinity;
    const x1ms = win ? win[1] : Infinity;
    const scored = names.map((name) => {
      const series = payload.metrics[name] || [];
      let sum = 0;
      let hasNonzero = false;
      for (let i = lo0; i <= hi0; i++) {
        const ms = parseTime(payload.t[i]);
        if (ms < x0ms || ms > x1ms) continue;
        const v = series[i];
        if (v != null && isFinite(v)) {
          if (v !== 0) hasNonzero = true;
          sum += v;
        }
      }
      return { name, sum, hasNonzero };
    });
    scored.sort((a, b) => b.sum - a.sum || a.name.localeCompare(b.name));
    const pinned = overlayMetricNames();
    const rest = scored
      .filter((x) => x.hasNonzero && !pinned.includes(x.name))
      .map((x) => x.name);
    return [...pinned, ...rest];
  }

  function renderMetricFilter() {
    if (!metricListEl) return;
    metricListEl.innerHTML = "";
    if (!payload) return;
    const ranked = metricsByViewSum();
    for (const name of ranked) {
      const lab = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = name;
      input.checked = visibleMetrics.has(name);
      input.addEventListener("change", () => {
        if (input.checked) visibleMetrics.add(name);
        else visibleMetrics.delete(name);
        onVisibleMetricsChanged({ keepY: false });
      });
      lab.appendChild(input);
      lab.appendChild(document.createTextNode(displayMetricName(name)));
      metricListEl.appendChild(lab);
    }
  }

  async function onVisibleMetricsChanged({ keepY = false } = {}) {
    if (keepY) {
      const bounds = currentYBounds();
      if (bounds && bounds[1] > bounds[0]) {
        yRange = [Number(bounds[0]), Number(bounds[1])];
        yAuto = false;
        syncYAutoButton();
      }
    } else {
      yAuto = true;
      syncYAutoButton();
    }
    const n = visibleMetrics.size;
    const active = metricsByViewSum().length;
    setStatus(
      keepY && n === 0
        ? "metric 전체 해제 (Y축 유지)"
        : `표시 metric ${n}/${active}`
    );
    renderMetricFilter();
    refreshHoverPanel();
    await draw();
  }

  function selectAllMetrics() {
    visibleMetrics = new Set(metricsByViewSum());
    onVisibleMetricsChanged({ keepY: false });
  }

  function clearAllMetrics() {
    visibleMetrics = new Set();
    onVisibleMetricsChanged({ keepY: true });
  }

  function isRateMetric(name) {
    return RATE_METRICS.has(name);
  }

  function rateMetricNames() {
    if (!payload?.metrics) return [];
    return [S_RATE_KEY, A_RATE_KEY].filter((k) => k in payload.metrics);
  }

  function overlayMetricNames() {
    const out = rateMetricNames();
    if (
      payload?.metrics?.[M971_COL] &&
      Array.isArray(payload.m971_tod_ref) &&
      payload.m971_tod_ref.length === payload.t?.length
    ) {
      out.push(M971_TOD_KEY);
    }
    return out;
  }

  function displayMetricName(name) {
    if (name === S_RATE_KEY) return "S_RATE (M658/M971)";
    if (name === A_RATE_KEY) return "A_RATE (M696/M971)";
    if (name === M971_TOD_KEY) return M971_TOD_LABEL;
    return name;
  }

  function ensureRateMetrics(data) {
    if (!data?.metrics?.[M971_COL]) return;
    const m971 = data.metrics[M971_COL];
    if (!data.metrics[S_RATE_KEY] && data.metrics.M658) {
      const m658 = data.metrics.M658;
      data.metrics[S_RATE_KEY] = m658.map((n, i) => {
        const d = m971[i];
        if (d == null || !isFinite(d) || d === 0) return null;
        const v = n / d;
        return isFinite(v) ? v : null;
      });
    }
    if (!data.metrics[A_RATE_KEY] && data.metrics.M696) {
      const m696 = data.metrics.M696;
      data.metrics[A_RATE_KEY] = m696.map((n, i) => {
        const d = m971[i];
        if (d == null || !isFinite(d) || d === 0) return null;
        const v = n / d;
        return isFinite(v) ? v : null;
      });
    }
  }

  function formatMetricValue(v, name) {
    if (v == null || !isFinite(v)) return "—";
    if (name && isRateMetric(name)) return Number(v).toFixed(3);
    return Math.round(v).toLocaleString("en-US");
  }

  function nearestTimeIndex(ms) {
    if (!payload?.t?.length || !isFinite(ms)) return null;
    const tms = (i) => parseTime(payload.t[i]);
    let a = 0;
    let b = payload.t.length - 1;
    while (a < b) {
      const m = (a + b) >> 1;
      if (tms(m) < ms) a = m + 1;
      else b = m;
    }
    let best = a;
    let bestD = Math.abs(tms(a) - ms);
    if (a > 0) {
      const d = Math.abs(tms(a - 1) - ms);
      if (d < bestD) {
        best = a - 1;
        bestD = d;
      }
    }
    if (a + 1 < payload.t.length) {
      const d = Math.abs(tms(a + 1) - ms);
      if (d < bestD) best = a + 1;
    }
    return best;
  }

  /** Midpoint of the current on-screen window (fallback: series center). */
  function defaultHoverIndex() {
    if (!payload?.t?.length) return null;
    const win = visibleXWindowMs();
    if (win && isFinite(win[0]) && isFinite(win[1]) && win[1] > win[0]) {
      const idx = nearestTimeIndex((win[0] + win[1]) / 2);
      if (idx != null) return idx;
    }
    return Math.floor((payload.t.length - 1) / 2);
  }

  function kstTodMinutes(tStr) {
    const ms = parseTime(tStr);
    if (!isFinite(ms)) return null;
    const d = new Date(ms);
    return d.getUTCHours() * 60 + d.getUTCMinutes();
  }

  /** Build m971_tod_ref on the client when missing (e.g. old top100 export). */
  function ensureM971TodRef(data) {
    if (!data?.t?.length || !data.metrics?.[M971_COL]) return;
    if (
      Array.isArray(data.m971_tod_ref) &&
      data.m971_tod_ref.length === data.t.length
    ) {
      return;
    }
    const sum = new Map();
    const cnt = new Map();
    for (let i = 0; i < data.t.length; i++) {
      const v = data.metrics[M971_COL][i];
      if (v == null || !isFinite(v)) continue;
      const tod = kstTodMinutes(data.t[i]);
      if (tod == null) continue;
      sum.set(tod, (sum.get(tod) || 0) + Number(v));
      cnt.set(tod, (cnt.get(tod) || 0) + 1);
    }
    data.m971_tod_ref = data.t.map((t, i) => {
      const tod = kstTodMinutes(t);
      if (tod == null) return null;
      const c = cnt.get(tod);
      return c ? sum.get(tod) / c : null;
    });
  }

  function m971TodRefEnabled(data) {
    return (
      visibleMetrics.has(M971_TOD_KEY) &&
      data?.metrics?.[M971_COL] &&
      Array.isArray(data.m971_tod_ref) &&
      data.m971_tod_ref.length === data.t?.length
    );
  }

  function refreshHoverPanel(index) {
    if (!hoverPanelEl) return;
    if (index == null) {
      index = interactionMode === "inspect" && inspectIndex != null
        ? inspectIndex
        : hoverIndex;
    }
    if (index == null) index = defaultHoverIndex();
    if (index == null || !payload?.t?.length) {
      hoverPanelEl.innerHTML = "<em>표시할 시점이 없습니다.</em>";
      return;
    }
    index = Math.max(0, Math.min(index, payload.t.length - 1));
    hoverIndex = index;
    const names = metricsByViewSum().filter((n) => visibleMetrics.has(n));
    if (!names.length) {
      hoverPanelEl.innerHTML =
        "<em>표시 중인 metric이 없습니다. 위에서 metric을 선택하세요.</em>";
      return;
    }
    const pairs = names
      .map((name) => {
        const v = payload.metrics[name]?.[index];
        return { name, val: v == null || !isFinite(v) ? NaN : Number(v) };
      })
      .filter((p) => isFinite(p.val) && p.val !== 0);
    if (m971TodRefEnabled(payload)) {
      const rv = payload.m971_tod_ref[index];
      const rval = rv == null || !isFinite(rv) ? NaN : Number(rv);
      if (isFinite(rval) && rval !== 0) {
        pairs.push({ name: M971_TOD_LABEL, val: rval });
      }
    }
    pairs.sort((a, b) => b.val - a.val);
    if (!pairs.length) {
      hoverPanelEl.innerHTML =
        "<em>이 시점에 0이 아닌 특성값이 없습니다.</em>";
      return;
    }
    const when = payload.t[index] || "";
    const cells = pairs
      .map((p, i) => {
        const bg = Math.floor(i / 4) % 2 ? "#f6f8fa" : "#ffffff";
        const shown = isFinite(p.val) ? formatMetricValue(p.val, p.name) : "—";
        return (
          `<div class="hover-cell" style="background:${bg}">` +
          `<span class="rank">${i + 1}</span>` +
          `<span class="name" title="${p.name}">${p.name}</span>` +
          `<span class="val">${shown}</span>` +
          `</div>`
        );
      })
      .join("");
    hoverPanelEl.innerHTML =
      `<div style="margin:0 0 6px;color:#666;font-size:0.78rem">${when}</div>` +
      `<div class="hover-grid">${cells}</div>`;
  }

  function graphHeightPx() {
    const h = graphEl?.clientHeight || 0;
    return Math.max(h > 40 ? h : 420, 260);
  }

  function resizePlot() {
    if (!graphEl || !graphEl.data) return;
    try {
      Plotly.Plots.resize(graphEl);
    } catch (_) {
      /* plot not ready */
    }
  }

  /** Mobile Safari often paints Plotly/WebGL before the container settles. */
  function scheduleResize() {
    if (_resizeTimer) clearTimeout(_resizeTimer);
    requestAnimationFrame(() => {
      resizePlot();
      requestAnimationFrame(() => {
        resizePlot();
        _resizeTimer = setTimeout(() => {
          resizePlot();
          _resizeTimer = setTimeout(resizePlot, 200);
        }, 60);
      });
    });
  }

  function syncYAutoButton() {
    const btn = document.getElementById("btn-y-auto");
    if (btn) btn.classList.toggle("active", yAuto);
  }

  function syncOverlayButtons() {
    document
      .getElementById("btn-mode-anomaly")
      ?.classList.toggle("active", showAnomalies);
    document
      .getElementById("btn-mode-plain")
      ?.classList.toggle("active", !showAnomalies);
  }

  function fillPlmnSelect(rows) {
    selectEl.innerHTML = "";
    for (const row of rows) {
      const opt = document.createElement("option");
      opt.value = row.plmn;
      const nLab = row.n_labels || 0;
      const lab = isLabeledCatalog() && nLab > 0 ? ` · labels ${nLab}` : "";
      opt.textContent = `#${String(row.rank ?? "").padStart(3, "0")} ${row.display}${lab}`;
      selectEl.appendChild(opt);
    }
  }

  function parseTime(v) {
    if (v == null) return NaN;
    if (typeof v === "number") return v;
    // Labels are UTC ISO; series t is KST-naive "YYYY-MM-DD HH:MM:SS".
    const s = String(v);
    if (s.includes("T") && (s.endsWith("Z") || /[+-]\d\d:\d\d$/.test(s))) {
      return Date.parse(s);
    }
    // Treat naive plot times as UTC slots (same convention as labeling app).
    const m = s.match(
      /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?/
    );
    if (m) {
      return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
    }
    return Date.parse(s);
  }

  function utcIsoToPlotNaive(iso) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    // Convert UTC instant → KST wall, encode as naive plot string.
    const kst = new Date(d.getTime() + 9 * 3600 * 1000);
    const pad = (n) => (n < 10 ? "0" : "") + n;
    return (
      kst.getUTCFullYear() +
      "-" +
      pad(kst.getUTCMonth() + 1) +
      "-" +
      pad(kst.getUTCDate()) +
      " " +
      pad(kst.getUTCHours()) +
      ":" +
      pad(kst.getUTCMinutes()) +
      ":" +
      pad(kst.getUTCSeconds())
    );
  }

  function msToPlotNaive(ms) {
    if (!isFinite(ms)) return null;
    const d = new Date(ms);
    const pad = (n) => (n < 10 ? "0" : "") + n;
    return (
      d.getUTCFullYear() +
      "-" +
      pad(d.getUTCMonth() + 1) +
      "-" +
      pad(d.getUTCDate()) +
      " " +
      pad(d.getUTCHours()) +
      ":" +
      pad(d.getUTCMinutes()) +
      ":" +
      pad(d.getUTCSeconds())
    );
  }

  /** Same KST wall format as labeling value-cursor tooltip. */
  function formatHoverTimeKst(ms) {
    if (!isFinite(ms)) return "";
    const d = new Date(ms);
    const pad = (n) => (n < 10 ? "0" : "") + n;
    return (
      d.getUTCFullYear() +
      "년 " +
      pad(d.getUTCMonth() + 1) +
      "월 " +
      pad(d.getUTCDate()) +
      "일 " +
      pad(d.getUTCHours()) +
      ":" +
      pad(d.getUTCMinutes())
    );
  }

  function edgeLine(x, color, width) {
    return {
      type: "line",
      xref: "x",
      yref: "paper",
      x0: x,
      x1: x,
      y0: 0,
      y1: 1,
      line: { color, width },
      layer: "above",
    };
  }

  function shapesForLabels(labels, highlightId) {
    const shapes = [];
    const hiId = highlightId == null ? null : String(highlightId);
    for (const item of labels || []) {
      const kind = (item.kind || "point").toLowerCase();
      const x0 = utcIsoToPlotNaive(item.start);
      const x1 = utcIsoToPlotNaive(item.end || item.start);
      const hi = hiId != null && String(item.id) === hiId;
      // Unselected = yellow, selected = crimson (thicker).
      const stroke = hi ? "crimson" : "#c9a227";
      const strokeW = hi ? 3 : 2;
      const isPoint = kind === "point" || item.start === item.end;

      if (isPoint) {
        shapes.push(edgeLine(x0, stroke, strokeW));
      } else {
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0,
          x1,
          y0: 0,
          y1: 1,
          fillcolor: hi ? "rgba(220,20,60,0.42)" : "rgba(201,162,39,0.28)",
          line: { width: 0 },
          layer: "above",
        });
        shapes.push(edgeLine(x0, stroke, strokeW));
        shapes.push(edgeLine(x1, stroke, strokeW));
      }
    }
    return shapes;
  }

  /** Nearest series envelope Y at plot time (for touchable anomaly markers). */
  function nearestEnvelopeY(ms) {
    if (!payload?.t?.length || !isFinite(ms)) return null;
    const tms = (i) => parseTime(payload.t[i]);
    let a = 0;
    let b = payload.t.length - 1;
    while (a < b) {
      const m = (a + b) >> 1;
      if (tms(m) < ms) a = m + 1;
      else b = m;
    }
    let best = a;
    let bestD = Math.abs(tms(a) - ms);
    if (a > 0) {
      const d = Math.abs(tms(a - 1) - ms);
      if (d < bestD) {
        best = a - 1;
        bestD = d;
      }
    }
    if (a + 1 < payload.t.length) {
      const d = Math.abs(tms(a + 1) - ms);
      if (d < bestD) best = a + 1;
    }
    let ymax = -Infinity;
    for (const name of allMetricNames()) {
      if (!visibleMetrics.has(name) || isRateMetric(name)) continue;
      const series = payload.metrics[name];
      const v = series?.[best];
      if (v != null && isFinite(v) && v > ymax) ymax = v;
    }
    return isFinite(ymax) ? ymax : null;
  }

  /** SVG markers (not WebGL) so mobile taps register via plotly_click. */
  function anomalyMarkerTrace(labels, highlightId) {
    const xs = [];
    const ys = [];
    const colors = [];
    const sizes = [];
    const ids = [];
    const baseSize = coarsePointer ? 18 : 11;
    const hiId = highlightId == null ? null : String(highlightId);
    for (const item of labels || []) {
      const kind = (item.kind || "point").toLowerCase();
      const edges =
        kind === "point" || item.start === item.end
          ? [utcIsoToPlotNaive(item.start)]
          : [
              utcIsoToPlotNaive(item.start),
              utcIsoToPlotNaive(item.end || item.start),
            ];
      const hi = hiId != null && String(item.id) === hiId;
      for (const x of edges) {
        const y = nearestEnvelopeY(parseTime(x));
        if (y == null) continue;
        xs.push(x);
        ys.push(y);
        colors.push(hi ? "crimson" : "#c9a227");
        sizes.push(hi ? baseSize + 5 : baseSize);
        ids.push(item.id);
      }
    }
    if (!xs.length) return null;
    return {
      type: "scatter",
      mode: "markers",
      name: "__anomaly_markers",
      x: xs,
      y: ys,
      customdata: ids,
      marker: {
        symbol: "x",
        size: sizes,
        color: colors,
        line: { width: coarsePointer ? 2.5 : 2, color: colors },
      },
      hoverinfo: "skip",
      showlegend: false,
      cliponaxis: false,
    };
  }

  /** Y auto from values visible in [x0ms, x1ms] (no forced zero). */
  function yRangeForWindow(data, x0ms, x1ms) {
    if (!data?.t?.length) return null;
    const names = allMetricNames().filter((n) => visibleMetrics.has(n));
    if (!names.length) return null;

    const bounds = windowIndexBounds(data, x0ms, x1ms);
    if (!bounds) return null;
    const [lo, hi] = bounds;

    let ymin = Infinity;
    let ymax = -Infinity;
    for (let i = lo; i <= hi; i++) {
      const ms = parseTime(data.t[i]);
      if (isFinite(x0ms) && ms < x0ms) continue;
      if (isFinite(x1ms) && ms > x1ms) continue;
      for (const name of names) {
        if (isRateMetric(name)) continue;
        const v = data.metrics[name]?.[i];
        if (v == null || !isFinite(v)) continue;
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
      }
      if (m971TodRefEnabled(data)) {
        const rv = data.m971_tod_ref[i];
        if (rv != null && isFinite(rv)) {
          if (rv < ymin) ymin = rv;
          if (rv > ymax) ymax = rv;
        }
      }
    }
    if (!(ymax >= ymin) || !isFinite(ymin) || !isFinite(ymax)) return null;
    const span = ymax - ymin;
    const pad = span > 0 ? span * 0.08 : Math.max(Math.abs(ymax) * 0.05, 1);
    let yLo = ymin - pad;
    let yHi = ymax + pad;
    // Don't open empty negative space when the visible series is non-negative.
    if (ymin >= 0) yLo = Math.max(0, yLo);
    return [yLo, yHi];
  }

  function syncYFromX(x0ms, x1ms) {
    if (!payload) return null;
    const yr = yRangeForWindow(payload, x0ms, x1ms);
    yRange = yr;
    return yr;
  }

  function visibleXWindowMs() {
    if (xRange) {
      return [parseTime(xRange[0]), parseTime(xRange[1])];
    }
    if (payload?.t?.length) {
      return [
        parseTime(payload.t[0]),
        parseTime(payload.t[payload.t.length - 1]),
      ];
    }
    const cur = currentXRangeMs();
    return cur;
  }

  function currentYBounds() {
    if (yRange && yRange.length === 2 && yRange[1] > yRange[0]) {
      return [Number(yRange[0]), Number(yRange[1])];
    }
    const r = graphEl._fullLayout?.yaxis?.range;
    if (r && Number(r[1]) > Number(r[0])) {
      return [Number(r[0]), Number(r[1])];
    }
    const win = visibleXWindowMs();
    if (!win) return null;
    return yRangeForWindow(payload, win[0], win[1]);
  }

  function applyYRelayout(yr) {
    if (!yr || !(yr[1] > yr[0])) return;
    yRange = [Number(yr[0]), Number(yr[1])];
    const update = {
      "yaxis.range": yRange,
      "yaxis.autorange": false,
    };
    // Pin X so a Y-only change never shifts the time window.
    const curX = currentXRangeMs();
    if (curX && curX[1] > curX[0]) {
      xRange = [msToPlotNaive(curX[0]), msToPlotNaive(curX[1])];
      update["xaxis.range"] = xRange;
      update["xaxis.autorange"] = false;
    } else if (xRange) {
      update["xaxis.range"] = xRange;
      update["xaxis.autorange"] = false;
    }
    suppressRelayout = true;
    Plotly.relayout(graphEl, update).then(() => {
      suppressRelayout = false;
    });
  }

  /** Keep the Y floor fixed; only the top moves (Y+ = 0.7, Y- = 1.4). */
  function scaleY(factor) {
    const bounds = currentYBounds();
    if (!bounds) return;
    let [lo, hi] = bounds;
    // Prefer an exact 0 floor when the visible series is non-negative.
    const win = visibleXWindowMs();
    const dataYr = win
      ? yRangeForWindow(payload, win[0], win[1])
      : null;
    if (yAuto && dataYr && dataYr[0] >= 0) lo = 0;
    else if (yAuto && lo >= 0) lo = 0;
    const top = lo + (hi - lo) * factor;
    if (!(top > lo)) return;
    yAuto = false;
    syncYAutoButton();
    applyYRelayout([lo, top]);
  }

  function resetYAuto() {
    yAuto = true;
    syncYAutoButton();
    const win = visibleXWindowMs();
    if (!win) return;
    const yr = syncYFromX(win[0], win[1]);
    applyYRelayout(yr);
  }

  function buildFigure(data, highlightId) {
    // Keep color order stable across toggles (full metric list order).
    const colorOrder = allMetricNames();
    const primaryNames = colorOrder.filter(
      (name) => visibleMetrics.has(name) && !isRateMetric(name),
    );
    const rateNames = colorOrder.filter(
      (name) => visibleMetrics.has(name) && isRateMetric(name),
    );
    const traces = primaryNames.map((name) => {
      const colorIdx = Math.max(0, colorOrder.indexOf(name));
      const color = SERIES_COLORWAY[colorIdx % SERIES_COLORWAY.length];
      return {
        type: "scattergl",
        mode: "lines",
        name,
        x: data.t,
        y: data.metrics[name],
        opacity: 0.72,
        line: { width: 1, color },
        hovertemplate:
          "<b>%{fullData.name}</b><br>" +
          "%{x|%Y년 %m월 %d일 %H:%M}<br>" +
          "값=%{y:,.0f}<extra></extra>",
        uid: name,
      };
    });

    for (const name of rateNames) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name,
        x: data.t,
        y: data.metrics[name],
        opacity: 0.9,
        yaxis: "y2",
        line: { ...REF_LINE, color: RATE_COLORS[name] || "#888" },
        hovertemplate:
          "<b>%{fullData.name}</b><br>" +
          "%{x|%Y년 %m월 %d일 %H:%M}<br>" +
          "값=%{y:.3f}<extra></extra>",
        uid: name,
      });
    }

    if (m971TodRefEnabled(data)) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: M971_TOD_LABEL,
        x: data.t,
        y: data.m971_tod_ref,
        line: { ...REF_LINE, color: M971_TOD_COLOR },
        hovertemplate:
          `<b>${M971_TOD_LABEL}</b><br>` +
          "%{x|%Y년 %m월 %d일 %H:%M}<br>" +
          "값=%{y:,.0f}<extra></extra>",
        uid: "__m971_tod_ref__",
      });
    }

    if (anomaliesEnabled()) {
      const markers = anomalyMarkerTrace(data.labels, highlightId);
      if (markers) traces.push(markers);
    }

    let x0ms = null;
    let x1ms = null;
    // Always pin X explicitly. With zero metric traces, Plotly autorange invents
    // a wrong date window that then "sticks" after metrics are re-selected.
    let xAxisRange = xRange;
    if (!xAxisRange && data.t?.length) {
      xAxisRange = [data.t[0], data.t[data.t.length - 1]];
    }
    if (xAxisRange) {
      x0ms = parseTime(xAxisRange[0]);
      x1ms = parseTime(xAxisRange[1]);
    }
    let yr = yRange;
    if (yAuto) {
      yr = syncYFromX(x0ms, x1ms);
    }

    const xaxis = {
      title: "시간 (KST)",
      type: "date",
      rangeslider: { visible: false },
      fixedrange: false,
      autorange: false,
      showspikes: false,
      hoverformat: "%Y년 %m월 %d일 %H:%M",
    };
    if (xAxisRange) xaxis.range = xAxisRange;

    const yaxis = {
      title: "value",
      fixedrange: true,
      autorange: !yr,
      rangemode: "normal",
    };
    if (yr) yaxis.range = yr;

    const layout = {
      margin: { l: 52, r: rateNames.length ? 55 : 20, t: 36, b: 48 },
      height: graphHeightPx(),
      autosize: true,
      showlegend: false,
      hovermode: "closest",
      dragmode,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#fffdf8",
      colorway: SERIES_COLORWAY.slice(),
      hoverlabel: {
        bgcolor: "white",
        font: { size: 11, family: "monospace" },
        align: "left",
        namelength: -1,
      },
      title: {
        text: isLabeledCatalog()
          ? `${data.display || data.plmn} · labels=${(data.labels || []).length}`
          : `${data.display || data.plmn}`,
        font: { size: 14 },
      },
      xaxis,
      yaxis,
      ...(rateNames.length
        ? {
            yaxis2: {
              title: "rate",
              overlaying: "y",
              side: "right",
              range: [0, 1],
              fixedrange: true,
              showgrid: false,
            },
          }
        : {}),
      shapes: (() => {
        const out = anomaliesEnabled()
          ? shapesForLabels(data.labels, highlightId)
          : [];
        const cursorIdx =
          interactionMode === "inspect" && inspectIndex != null
            ? inspectIndex
            : null;
        if (cursorIdx != null && data.t?.[cursorIdx] != null) {
          const x = data.t[cursorIdx];
          out.push({
            type: "line",
            xref: "x",
            yref: "paper",
            x0: x,
            x1: x,
            y0: 0,
            y1: 1,
            line: { color: "royalblue", width: 2, dash: "dot" },
            layer: "above",
            name: "value_cursor",
          });
        }
        return out;
      })(),
      annotations: (() => {
        const cursorIdx =
          interactionMode === "inspect" && inspectIndex != null
            ? inspectIndex
            : null;
        if (cursorIdx == null || data.t?.[cursorIdx] == null) return [];
        const x = data.t[cursorIdx];
        const ms = parseTime(x);
        return [
          {
            x,
            y: 1,
            xref: "x",
            yref: "paper",
            text: `값 탐색 · ${formatHoverTimeKst(ms)}`,
            showarrow: false,
            yshift: -8,
            font: { size: 11, color: "white" },
            bgcolor: "royalblue",
            borderpad: 3,
            name: "value_cursor",
          },
        ];
      })(),
    };

    return { data: traces, layout };
  }

  let _drawGen = 0;

  async function draw() {
    if (!payload) return;
    const gen = ++_drawGen;
    suppressRelayout = true;
    const fig = buildFigure(payload, selectedId);
    await Plotly.react(graphEl, fig.data, fig.layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      showTips: false,
      // Help mobile: scroll parent, pan plot
      scrollZoom: false,
    });
    // A newer draw (or inspect stepping) started while we were awaiting react.
    if (gen !== _drawGen) return;
    // scattergl often keeps a stale viewport until an explicit resize + Y pin.
    const win = visibleXWindowMs();
    let yr = yRange;
    if (yAuto && win) yr = syncYFromX(win[0], win[1]);
    const post = { height: graphHeightPx(), dragmode };
    if (yr && yr[1] > yr[0]) {
      post["yaxis.range"] = yr;
      post["yaxis.autorange"] = false;
    }
    // Re-pin X after react (empty-trace draws must not leave a bogus autorange).
    const xr =
      xRange ||
      (payload.t?.length
        ? [payload.t[0], payload.t[payload.t.length - 1]]
        : null);
    if (xr) {
      post["xaxis.range"] = xr;
      post["xaxis.autorange"] = false;
    }
    await Plotly.relayout(graphEl, post);
    if (gen !== _drawGen) return;
    suppressRelayout = false;
    bindAnomalyShapeClicks();
    bindInspectClicks();
    installPlaceTimeTip();
    // Re-paint inspect cursor from the live index (react may have used a stale snapshot).
    if (interactionMode === "inspect" && inspectIndex != null) {
      syncInspectCursorShape();
    }
    scheduleResize();
  }

  /** Empty-area time tip + royalblue dotted spike (same as labeling app). */
  function installPlaceTimeTip() {
    if (!graphEl || _placeTipBound) return;
    _placeTipBound = true;
    let raf = null;
    let lastText = "";
    let lastSpikeX = null;

    function ensureTip() {
      let tip = document.getElementById("place-time-tip");
      if (!tip) {
        tip = document.createElement("div");
        tip.id = "place-time-tip";
        tip.style.cssText = [
          "position:fixed",
          "z-index:99999",
          "pointer-events:none",
          "display:none",
          "background:#fff",
          "color:#444",
          "padding:6px 8px",
          "border:1px solid #bbb",
          "border-radius:2px",
          "font:11px/1.4 monospace",
          "white-space:nowrap",
          "box-shadow:0 1px 3px rgba(0,0,0,.18)",
        ].join(";");
        document.body.appendChild(tip);
      }
      return tip;
    }

    function ensureSpike() {
      let spike = document.getElementById("place-time-spike");
      if (!spike) {
        spike = document.createElement("div");
        spike.id = "place-time-spike";
        spike.style.cssText = [
          "position:fixed",
          "z-index:99998",
          "pointer-events:none",
          "display:none",
          "width:0",
          "border-left:1px dotted royalblue",
          "box-sizing:border-box",
        ].join(";");
        document.body.appendChild(spike);
      }
      return spike;
    }

    function hideTip() {
      const tip = document.getElementById("place-time-tip");
      if (tip) tip.style.display = "none";
      const spike = document.getElementById("place-time-spike");
      if (spike) spike.style.display = "none";
      lastText = "";
      lastSpikeX = null;
    }

    function formatHoverTime(ms) {
      const d = new Date(ms);
      const pad = (n) => (n < 10 ? "0" : "") + n;
      return (
        d.getUTCFullYear() +
        "년 " +
        pad(d.getUTCMonth() + 1) +
        "월 " +
        pad(d.getUTCDate()) +
        "일 " +
        pad(d.getUTCHours()) +
        ":" +
        pad(d.getUTCMinutes())
      );
    }

    function snapFiveMinMs(ms) {
      if (!isFinite(ms)) return NaN;
      const step = 5 * 60 * 1000;
      return Math.round(ms / step) * step;
    }

    function nearestSampleMs(ms) {
      if (!payload?.t?.length || !isFinite(ms)) return NaN;
      const idx = nearestTimeIndex(ms);
      if (idx == null) return snapFiveMinMs(ms);
      return parseTime(payload.t[idx]);
    }

    graphEl.on("plotly_hover", () => {
      _placeOnTrace = true;
      const tip = document.getElementById("place-time-tip");
      if (tip) tip.style.display = "none";
      lastText = "";
    });
    graphEl.on("plotly_unhover", () => {
      _placeOnTrace = false;
    });

    graphEl.addEventListener(
      "pointermove",
      (ev) => {
        if (ev.buttons) {
          hideTip();
          return;
        }
        const layer =
          graphEl.querySelector(".nsewdrag") || graphEl;
        const bb = layer.getBoundingClientRect();
        if (
          ev.clientX < bb.left ||
          ev.clientX > bb.right ||
          ev.clientY < bb.top ||
          ev.clientY > bb.bottom
        ) {
          hideTip();
          return;
        }
        const ms = clientXToPlotMs(ev.clientX);
        if (!isFinite(ms)) {
          hideTip();
          return;
        }
        const snapped = nearestSampleMs(ms);
        if (!isFinite(snapped)) {
          hideTip();
          return;
        }
        // spikesnap=cursor: follow pointer. Data-snapped d2p drifts far from the
        // mouse after zoom (5-min sample spacing becomes many pixels).
        const left = Math.min(bb.right, Math.max(bb.left, ev.clientX));
        if (lastSpikeX !== left) {
          lastSpikeX = left;
          const spike = ensureSpike();
          spike.style.left = left + "px";
          spike.style.top = bb.top + "px";
          spike.style.height = Math.max(0, bb.height) + "px";
          spike.style.display = "block";
        }
        // On a series, Plotly hoverlabel already shows the time.
        if (_placeOnTrace) return;
        const text = formatHoverTime(snapped);
        if (
          text === lastText &&
          document.getElementById("place-time-tip")?.style.display === "block"
        ) {
          const tip0 = document.getElementById("place-time-tip");
          tip0.style.left = ev.clientX + 14 + "px";
          tip0.style.top = Math.max(8, ev.clientY - 32) + "px";
          return;
        }
        lastText = text;
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const tip = ensureTip();
          tip.textContent = text;
          tip.style.display = "block";
          let tipLeft = ev.clientX + 14;
          let tipTop = ev.clientY - 32;
          const w = tip.offsetWidth || 160;
          if (tipLeft + w > window.innerWidth - 8) tipLeft = ev.clientX - w - 14;
          if (tipTop < 8) tipTop = ev.clientY + 18;
          tip.style.left = tipLeft + "px";
          tip.style.top = tipTop + "px";
        });
      },
      true
    );

    graphEl.addEventListener("pointerleave", hideTip, true);
  }

  /** Map a mouse click in the plot area to data-X (ms). */
  function clientXToPlotMs(clientX) {
    const full = graphEl._fullLayout;
    const xa = full?.xaxis;
    if (!xa) return NaN;
    const layer =
      graphEl.querySelector(".nsewdrag") ||
      graphEl.querySelector(".plotly .main-svg");
    if (!layer) return NaN;
    const bb = layer.getBoundingClientRect();
    const px = clientX - bb.left;
    if (px < 0 || px > bb.width) return NaN;
    try {
      // Plotly date axis: p2d(pixel-from-plot-left) → Date / number
      if (typeof xa.p2d === "function") return parseTime(xa.p2d(px));
      if (typeof xa.p2c === "function" && typeof xa.c2d === "function") {
        return parseTime(xa.c2d(xa.p2c(px)));
      }
    } catch (_) {
      /* fall through */
    }
    // Fallback: linear map using current range.
    const cur = currentXRangeMs();
    if (!cur || !(cur[1] > cur[0]) || !(bb.width > 0)) return NaN;
    return cur[0] + (px / bb.width) * (cur[1] - cur[0]);
  }

  function labelAtPlotTime(ts, forTouch) {
    if (!isFinite(ts) || !payload?.labels?.length) return null;
    const win = currentXRangeMs();
    const winW =
      win && win[1] > win[0] ? win[1] - win[0] : 24 * 3600 * 1000;
    // Touch needs a wider hit area than mouse.
    const frac = forTouch || coarsePointer ? 0.035 : 0.012;
    const minTol = forTouch || coarsePointer ? 45 * 60 * 1000 : 15 * 60 * 1000;
    const lineTol = Math.max(minTol, winW * frac);

    let best = null;
    let bestScore = Infinity;
    for (const item of payload.labels) {
      const a = parseTime(utcIsoToPlotNaive(item.start));
      const b = parseTime(utcIsoToPlotNaive(item.end || item.start));
      const kind = (item.kind || "point").toLowerCase();
      const edges =
        kind === "point" || item.start === item.end ? [a] : [a, b];
      for (const edge of edges) {
        const d = Math.abs(ts - edge);
        if (d <= lineTol && d < bestScore) {
          bestScore = d;
          best = item;
        }
      }
    }
    if (best) return best;

    // Inside a range fill (secondary).
    for (const item of payload.labels) {
      const kind = (item.kind || "point").toLowerCase();
      if (kind === "point" || item.start === item.end) continue;
      const a = parseTime(utcIsoToPlotNaive(item.start));
      const b = parseTime(utcIsoToPlotNaive(item.end || item.start));
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      if (ts >= lo && ts <= hi) {
        const span = hi - lo;
        if (span < bestScore) {
          bestScore = span;
          best = item;
        }
      }
    }
    return best;
  }

  let _ptrDown = null;
  let _tapLockUntil = 0;
  function bindAnomalyShapeClicks() {
    const layer = graphEl.querySelector(".nsewdrag");
    if (!layer) return;
    if (layer.__anomalyClickBound) return;
    layer.__anomalyClickBound = true;

    layer.addEventListener(
      "pointerdown",
      (ev) => {
        _ptrDown = {
          x: ev.clientX,
          y: ev.clientY,
          t: Date.now(),
          type: ev.pointerType || "mouse",
        };
      },
      { passive: true }
    );

    const handleTap = (clientX, clientY, pointerType) => {
      if (!anomaliesEnabled() || !payload) return false;
      if (Date.now() < _tapLockUntil) return false;
      if (!_ptrDown) return false;
      const dist = Math.hypot(clientX - _ptrDown.x, clientY - _ptrDown.y);
      const dt = Date.now() - _ptrDown.t;
      const isTouch =
        pointerType === "touch" ||
        pointerType === "pen" ||
        _ptrDown.type === "touch" ||
        _ptrDown.type === "pen" ||
        coarsePointer;
      const maxDist = isTouch ? 24 : 8;
      const maxDt = isTouch ? 500 : 700;
      if (dist > maxDist || dt > maxDt) return false;
      const ts = clientXToPlotMs(clientX);
      const hit = labelAtPlotTime(ts, isTouch);
      _ptrDown = null;
      if (!hit) return false;
      _tapLockUntil = Date.now() + 350;
      toggleLabelFromGraph(hit.id);
      return true;
    };

    layer.addEventListener("pointerup", (ev) => {
      if (handleTap(ev.clientX, ev.clientY, ev.pointerType)) {
        ev.preventDefault();
        ev.stopPropagation();
      }
    });

    // Some mobile browsers fire touchend without a reliable click.
    layer.addEventListener(
      "touchend",
      (ev) => {
        const t = ev.changedTouches?.[0];
        if (!t) return;
        if (handleTap(t.clientX, t.clientY, "touch")) {
          ev.preventDefault();
          ev.stopPropagation();
        }
      },
      { passive: false }
    );

    layer.addEventListener("click", (ev) => {
      if (Date.now() < _tapLockUntil) {
        ev.preventDefault();
        ev.stopPropagation();
        return;
      }
      handleTap(ev.clientX, ev.clientY, "mouse");
    });
  }

  function dataXBoundsMs() {
    if (!payload?.t?.length) return null;
    const tmin = parseTime(payload.t[0]);
    const tmax = parseTime(payload.t[payload.t.length - 1]);
    if (!(tmax > tmin)) return null;
    return [tmin, tmax];
  }

  /** Keep [a,b] inside the dataset; preserve width when possible. */
  function clampTimeWindow(a, b) {
    const bounds = dataXBoundsMs();
    if (!bounds || !(b > a)) return [a, b];
    const [tmin, tmax] = bounds;
    let na = a;
    let nb = b;
    const width = nb - na;
    const full = tmax - tmin;
    if (width >= full) return [tmin, tmax];
    if (na < tmin) {
      na = tmin;
      nb = tmin + width;
    }
    if (nb > tmax) {
      nb = tmax;
      na = tmax - width;
    }
    if (na < tmin) na = tmin;
    return [na, nb];
  }

  function applyXYRelayout(x0ms, x1ms, { checkOffscreen = false } = {}) {
    if (!(x1ms > x0ms)) return;
    [x0ms, x1ms] = clampTimeWindow(x0ms, x1ms);
    if (!(x1ms > x0ms)) return;
    xRange = [msToPlotNaive(x0ms), msToPlotNaive(x1ms)];
    const update = {
      "xaxis.range": xRange,
      "xaxis.autorange": false,
      "yaxis.autorange": false,
    };
    if (yAuto) {
      const yr = syncYFromX(x0ms, x1ms);
      if (yr) update["yaxis.range"] = yr;
    } else if (yRange) {
      update["yaxis.range"] = yRange;
    }
    suppressRelayout = true;
    Plotly.relayout(graphEl, update).then(async () => {
      suppressRelayout = false;
      // After drag/button zoom-in: drop selection if it left the view.
      if (checkOffscreen) await clearHighlightIfOffscreen();
      renderMetricFilter();
      refreshHoverPanel();
    });
  }

  function labelOverlapsView(item, lo, hi) {
    const a = parseTime(utcIsoToPlotNaive(item.start));
    const b = parseTime(utcIsoToPlotNaive(item.end || item.start));
    if (!isFinite(a) || !isFinite(b) || !(hi > lo)) return false;
    const s = Math.min(a, b);
    const e = Math.max(a, b);
    return s <= hi && e >= lo;
  }

  async function clearHighlightIfOffscreen() {
    if (selectedId == null || !payload?.labels?.length) return false;
    const item = payload.labels.find((x) => String(x.id) === String(selectedId));
    const cur = currentXRangeMs();
    if (!item || !cur) {
      await clearLabelHighlight();
      return true;
    }
    if (labelOverlapsView(item, cur[0], cur[1])) return false;
    await clearLabelHighlight();
    return true;
  }

  let _labelListSyncing = false;

  function renderLabelList() {
    if (!listEl) return;
    const labels = payload?.labels || [];
    const prev = selectedId == null ? "" : String(selectedId);
    _labelListSyncing = true;
    listEl.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = labels.length
      ? "anomaly 구간 선택"
      : "라벨 없음";
    listEl.appendChild(empty);
    for (const item of labels) {
      const opt = document.createElement("option");
      opt.value = String(item.id);
      opt.textContent = item.line || `${item.kind} ${item.id}`;
      listEl.appendChild(opt);
    }
    const keep = prev && labels.some((x) => String(x.id) === prev);
    listEl.value = keep ? prev : "";
    _labelListSyncing = false;
  }

  async function clearLabelHighlight() {
    selectedId = null;
    _labelListSyncing = true;
    if (listEl) listEl.value = "";
    _labelListSyncing = false;
    setStatus("라벨 선택이 해제되었습니다.");
    await draw();
  }

  async function clearLabelSelection() {
    selectedId = null;
    _labelListSyncing = true;
    if (listEl) listEl.value = "";
    _labelListSyncing = false;
    setStatus("라벨 선택이 해제되었습니다.");
    // Same as the toolbar 「전체」 button (목록 「선택 해제」만).
    await resetX();
  }

  async function zoomSelectedLabel() {
    if (selectedId == null) {
      setStatus("선택된 anomaly가 없습니다.");
      return;
    }
    await selectLabel(selectedId, true);
  }

  async function toggleLabelFromGraph(id) {
    if (
      id != null &&
      selectedId != null &&
      String(selectedId) === String(id)
    ) {
      await clearLabelHighlight();
      return;
    }
    await selectLabel(id, false);
  }

  async function selectLabel(id, zoomTo) {
    if (id == null || id === "") {
      // Keep zoom; only the 「선택 해제」 button resets to full.
      await clearLabelHighlight();
      return;
    }
    const sid = String(id);
    const item = (payload.labels || []).find((x) => String(x.id) === sid);
    selectedId = item ? item.id : sid;
    renderLabelList();
    if (item && zoomTo) {
      const a = parseTime(utcIsoToPlotNaive(item.start));
      const b = parseTime(utcIsoToPlotNaive(item.end || item.start));
      let lo = Math.min(a, b);
      let hi = Math.max(a, b);
      const kind = (item.kind || "point").toLowerCase();
      const isPoint = kind === "point" || item.start === item.end || lo === hi;
      // Point: fixed window = 30 min × (1/0.7)^9 (two more zoom-outs than ^7).
      // Range: tight fit (min 30 min) then × (1/0.7)^7.
      const minTight = 30 * 60 * 1000;
      const widenRange = Math.pow(1 / 0.7, 7);
      const widenPoint = Math.pow(1 / 0.7, 9);
      const fixedPoint = minTight * widenPoint;
      let wide;
      if (isPoint) {
        wide = fixedPoint;
      } else {
        wide = Math.max(hi - lo, minTight) * widenRange;
      }
      const mid = (lo + hi) / 2;
      const half = wide / 2;
      lo = mid - half;
      hi = mid + half;
      [lo, hi] = clampTimeWindow(lo, hi);
      xRange = [msToPlotNaive(lo), msToPlotNaive(hi)];
      yAuto = true;
      syncYAutoButton();
      // Compute Y before draw so both react + post-relayout share the same range.
      const yr = syncYFromX(lo, hi);
      // 「선택 구간으로 줌」 → 이동 (Y auto). Apply before draw so layout.dragmode is pan.
      setInteractionMode("pan", { redraw: false });
      setStatus(`선택 구간으로 줌: ${item.line || item.id}`);
      await draw();
      suppressRelayout = true;
      try {
        const patch = {
          "xaxis.range": xRange,
          "xaxis.autorange": false,
          dragmode: "pan",
          height: graphHeightPx(),
        };
        if (yr) {
          patch["yaxis.range"] = yr;
          patch["yaxis.autorange"] = false;
        }
        await Plotly.relayout(graphEl, patch);
      } finally {
        suppressRelayout = false;
      }
      scheduleResize();
      renderMetricFilter();
      refreshHoverPanel();
    } else if (item) {
      setStatus(`선택: ${item.line || item.id}`);
      await draw();
    } else {
      await draw();
    }
  }

  async function setShowAnomalies(on) {
    if (!isLabeledCatalog()) return;
    if (showAnomalies === on) return;
    showAnomalies = on;
    syncOverlayButtons();
    await draw();
    setStatus(on ? "anomaly 표시" : "anomaly 숨김");
  }

  function setInteractionMode(mode, { redraw = true } = {}) {
    interactionMode = mode;
    dragmode = mode === "inspect" ? false : mode;
    document
      .getElementById("btn-pan")
      ?.classList.toggle("active", mode === "pan");
    document
      .getElementById("btn-zoom")
      ?.classList.toggle("active", mode === "zoom");
    document
      .getElementById("btn-inspect")
      ?.classList.toggle("active", mode === "inspect");
    if (mode !== "inspect") {
      inspectIndex = null;
    } else if (inspectIndex == null && hoverIndex != null) {
      inspectIndex = hoverIndex;
    } else if (inspectIndex == null) {
      inspectIndex = defaultHoverIndex();
    }
    if (graphEl && graphEl.data) {
      Plotly.relayout(graphEl, { dragmode: dragmode });
    }
    if (!redraw) return;
    draw().then(() => {
      refreshHoverPanel(
        interactionMode === "inspect" ? inspectIndex : hoverIndex
      );
      if (mode === "inspect") {
        setStatus("값 탐색: 클릭 또는 ←/→ · 드래그 이동 없음");
      }
    });
  }

  /** Move inspect cursor via shapes/annotations relayout only — never full Plotly.react. */
  let _inspectCursorBusy = false;
  let _inspectCursorNeedsSync = false;

  function syncInspectCursorShape() {
    if (!graphEl?.layout || !payload?.t?.length) return;
    if (_inspectCursorBusy) {
      _inspectCursorNeedsSync = true;
      return;
    }
    const shapes = anomaliesEnabled()
      ? shapesForLabels(payload.labels, selectedId)
      : [];
    let annotations = (graphEl.layout.annotations || []).filter(
      (a) => a && a.name !== "value_cursor"
    );
    if (
      interactionMode === "inspect" &&
      inspectIndex != null &&
      payload.t[inspectIndex] != null
    ) {
      const x = payload.t[inspectIndex];
      const ms = parseTime(x);
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color: "royalblue", width: 2, dash: "dot" },
        layer: "above",
        name: "value_cursor",
      });
      annotations = [
        ...annotations,
        {
          x,
          y: 1,
          xref: "x",
          yref: "paper",
          text: `값 탐색 · ${formatHoverTimeKst(ms)}`,
          showarrow: false,
          yshift: -8,
          font: { size: 11, color: "white" },
          bgcolor: "royalblue",
          borderpad: 3,
          name: "value_cursor",
        },
      ];
    }
    _inspectCursorBusy = true;
    _inspectCursorNeedsSync = false;
    suppressRelayout = true;
    Plotly.relayout(graphEl, { shapes, annotations })
      .catch(() => {})
      .finally(() => {
        _inspectCursorBusy = false;
        suppressRelayout = false;
        if (_inspectCursorNeedsSync) syncInspectCursorShape();
      });
  }

  function selectInspectAtMs(ms) {
    if (!payload?.t?.length || !isFinite(ms)) return;
    const idx = nearestTimeIndex(ms);
    if (idx == null) return;
    _drawGen += 1;
    inspectIndex = idx;
    hoverIndex = idx;
    refreshHoverPanel(idx);
    setStatus(`값 탐색: ${payload.t[idx]} · ←/→ 키로 이동`);
    syncInspectCursorShape();
  }

  let _inspectStepPending = 0;
  let _inspectStepRaf = null;
  let _inspectPanelTimer = null;

  function applyInspectStep(delta) {
    if (interactionMode !== "inspect" || !payload?.t?.length || !delta) return;
    // Invalidate in-flight full draws so a stale Plotly.react cannot snap the cursor back.
    _drawGen += 1;
    let idx = inspectIndex;
    if (idx == null) idx = hoverIndex;
    if (idx == null) idx = defaultHoverIndex();
    if (idx == null) return;
    idx = Math.max(0, Math.min(payload.t.length - 1, idx + delta));
    inspectIndex = idx;
    hoverIndex = idx;
    setStatus(`값 탐색: ${payload.t[idx]} · ←/→ 키로 이동`);
    syncInspectCursorShape();
    // Debounce the heavy hover-panel DOM rebuild while a key is held.
    clearTimeout(_inspectPanelTimer);
    _inspectPanelTimer = setTimeout(() => refreshHoverPanel(inspectIndex), 40);
  }

  function stepInspect(delta) {
    // Coalesce key-repeat into one move per animation frame.
    _inspectStepPending += delta;
    if (_inspectStepRaf != null) return;
    _inspectStepRaf = requestAnimationFrame(() => {
      _inspectStepRaf = null;
      const d = _inspectStepPending;
      _inspectStepPending = 0;
      applyInspectStep(d);
    });
  }

  function bindInspectClicks() {
    const layer = graphEl.querySelector(".nsewdrag");
    if (!layer || layer.__inspectClickBound) return;
    layer.__inspectClickBound = true;
    let ptr = null;
    layer.addEventListener(
      "pointerdown",
      (ev) => {
        ptr = { x: ev.clientX, y: ev.clientY };
      },
      { passive: true }
    );
    layer.addEventListener("click", (ev) => {
      if (interactionMode !== "inspect") return;
      if (ptr && Math.hypot(ev.clientX - ptr.x, ev.clientY - ptr.y) > 10) return;
      const ms = clientXToPlotMs(ev.clientX);
      if (!isFinite(ms)) return;
      selectInspectAtMs(ms);
    });
  }

  function setDragMode(mode) {
    // Back-compat alias used by older callers.
    setInteractionMode(mode);
  }

  function currentXRangeMs() {
    const layout = graphEl._fullLayout;
    if (xRange) {
      return [parseTime(xRange[0]), parseTime(xRange[1])];
    }
    if (!layout?.xaxis?.range) {
      if (payload?.t?.length) {
        return [
          parseTime(payload.t[0]),
          parseTime(payload.t[payload.t.length - 1]),
        ];
      }
      return null;
    }
    const r = layout.xaxis.range;
    return [parseTime(r[0]), parseTime(r[1])];
  }

  function isFullXView() {
    // null xRange means 「전체」. Also treat near-full windows as full.
    if (xRange == null) return true;
    if (!payload?.t?.length) return true;
    const full0 = parseTime(payload.t[0]);
    const full1 = parseTime(payload.t[payload.t.length - 1]);
    const cur = currentXRangeMs();
    if (!cur || !(full1 > full0) || !(cur[1] > cur[0])) return true;
    const span = Math.max(full1 - full0, cur[1] - cur[0], 1);
    return (
      Math.abs(cur[0] - full0) / span < 0.02 &&
      Math.abs(cur[1] - full1) / span < 0.02
    );
  }

  function scaleX(factor, fromLeft) {
    const cur = currentXRangeMs();
    if (!cur) return;
    let [a, b] = cur;
    if (!(b > a)) return;
    const width = b - a;
    if (fromLeft) {
      const nw = Math.max(width * factor, 30 * 60 * 1000);
      b = a + nw;
    } else {
      const mid = (a + b) / 2;
      const half = Math.max((width * factor) / 2, 15 * 60 * 1000);
      a = mid - half;
      b = mid + half;
    }
    applyXYRelayout(a, b, { checkOffscreen: factor < 1 });
  }

  async function resetX() {
    xRange = null;
    yRange = null;
    yAuto = true;
    syncYAutoButton();
    await draw();
    renderMetricFilter();
    refreshHoverPanel();
  }

  async function loadPlmn(plmn) {
    setStatus("로딩 중…");
    selectedId = null;
    xRange = null;
    yRange = null;
    yAuto = true;
    hoverIndex = null;
    inspectIndex = null;
    syncYAutoButton();
    const res = await fetch(`${dataBase}/${plmn}.json`, { cache: "no-store" });
    if (!res.ok) throw new Error(`failed to load ${plmn}`);
    payload = await res.json();
    ensureRateMetrics(payload);
    ensureM971TodRef(payload);
    visibleMetrics = new Set([...allMetricNames(), ...overlayMetricNames()]);
    hoverIndex = defaultHoverIndex();
    if (interactionMode === "inspect") inspectIndex = hoverIndex;
    renderMetricFilter();
    if (isLabeledCatalog()) renderLabelList();
    await draw();
    refreshHoverPanel(hoverIndex);
    setStatus(
      isLabeledCatalog()
        ? `${payload.start_kst} ~ ${payload.end_kst} · ${payload.n_points} pts · ${
            (payload.labels || []).length
          } labels`
        : `${payload.start_kst} ~ ${payload.end_kst} · ${payload.n_points} pts`
    );
  }

  function onRelayout(ev) {
    if (suppressRelayout || !payload || !ev) return;
    const touchedX =
      ev["xaxis.range[0]"] != null ||
      ev["xaxis.range[1]"] != null ||
      ev["xaxis.range"] != null ||
      ev["xaxis.autorange"] === true;
    if (!touchedX) return;

    if (ev["xaxis.autorange"] === true) {
      xRange = null;
      if (yAuto) yRange = null;
      draw().then(() => {
        renderMetricFilter();
        refreshHoverPanel();
      });
      return;
    }

    // Prefer the event / live layout — currentXRangeMs() would return the
    // stale stored xRange and undo the user's drag zoom.
    let x0 = null;
    let x1 = null;
    if (ev["xaxis.range"] != null) {
      x0 = parseTime(ev["xaxis.range"][0]);
      x1 = parseTime(ev["xaxis.range"][1]);
    } else if (
      ev["xaxis.range[0]"] != null ||
      ev["xaxis.range[1]"] != null
    ) {
      const r = graphEl._fullLayout?.xaxis?.range;
      if (r) {
        x0 = parseTime(r[0]);
        x1 = parseTime(r[1]);
      }
    }
    if (x0 == null || x1 == null || !(x1 > x0)) return;
    // Reject bogus windows from empty-plot autorange (no overlap with data).
    if (payload.t?.length) {
      const d0 = parseTime(payload.t[0]);
      const d1 = parseTime(payload.t[payload.t.length - 1]);
      if (!(d1 > d0)) return;
      if (x1 < d0 || x0 > d1) return;
    }
    applyXYRelayout(x0, x1, { checkOffscreen: true });
  }

  async function loadCatalog() {
    const meta = CATALOGS[activeCatalog] || CATALOGS.labeled;
    const res = await fetch(`${dataBase}/index.json`, { cache: "no-store" });
    if (!res.ok) {
      setStatus(meta.emptyIndex);
      return;
    }
    const idx = await res.json();
    catalog = idx.plmns || [];
    syncOverlayButtons();
    fillPlmnSelect(catalog);
    if (!catalog.length) {
      setStatus("보낼 사업자가 없습니다.");
      return;
    }
    await loadPlmn(catalog[0].plmn);
  }

  async function init() {
    activeCatalog = catalogFromHash();
    dataBase = CATALOGS[activeCatalog].dataBase;
    syncCatalogTabs();
    tabLabeledEl.addEventListener("click", () => switchCatalog("labeled"));
    tabTop100El.addEventListener("click", () => switchCatalog("top100"));
    window.addEventListener("hashchange", () => {
      const next = catalogFromHash();
      if (next !== activeCatalog) switchCatalog(next);
    });

    selectEl.addEventListener("change", () => loadPlmn(selectEl.value));
    listEl.addEventListener("change", () => {
      if (_labelListSyncing) return;
      const id = listEl.value;
      if (!id) {
        // Clear highlight only; keep current zoom window.
        // 「선택 해제」 button is what resets to full view.
        clearLabelHighlight();
        return;
      }
      // Criterion is view state only (not whether a label is selected).
      selectLabel(id, !isFullXView());
    });
    document.getElementById("btn-clear-selection").onclick = () =>
      clearLabelSelection();
    document.getElementById("btn-zoom-selected").onclick = () =>
      zoomSelectedLabel();
    document.getElementById("btn-pan").onclick = () =>
      setInteractionMode("pan");
    document.getElementById("btn-zoom").onclick = () =>
      setInteractionMode("zoom");
    document.getElementById("btn-inspect").onclick = () =>
      setInteractionMode("inspect");
    document.getElementById("btn-zoom-in").onclick = () => scaleX(0.7, false);
    document.getElementById("btn-zoom-out").onclick = () => scaleX(1 / 0.7, false);
    document.getElementById("btn-reset").onclick = async () => {
      // Switch to 줌 before redraw so layout.dragmode sticks (same idea as
      // 「선택 구간으로 줌」 → 이동).
      setInteractionMode("zoom", { redraw: false });
      await resetX();
      if (graphEl?.data) {
        suppressRelayout = true;
        try {
          await Plotly.relayout(graphEl, { dragmode: "zoom" });
        } finally {
          suppressRelayout = false;
        }
      }
    };
    document.getElementById("btn-y-in").onclick = () => scaleY(0.7);
    document.getElementById("btn-y-out").onclick = () => scaleY(1.4);
    document.getElementById("btn-y-auto").onclick = () => resetYAuto();
    document.getElementById("btn-mode-anomaly").onclick = () =>
      setShowAnomalies(true);
    document.getElementById("btn-mode-plain").onclick = () =>
      setShowAnomalies(false);
    document.getElementById("btn-metrics-all").onclick = () =>
      selectAllMetrics();
    document.getElementById("btn-metrics-none").onclick = () =>
      clearAllMetrics();
    syncYAutoButton();

    window.addEventListener("keydown", (ev) => {
      if (interactionMode !== "inspect") return;
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        stepInspect(-1);
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        stepInspect(1);
      }
    });

    window.addEventListener("resize", scheduleResize);
    window.addEventListener("orientationchange", () => {
      // Wait for viewport to settle after rotate, then full redraw.
      setTimeout(async () => {
        await draw();
        scheduleResize();
      }, 280);
    });
    if (window.visualViewport) {
      window.visualViewport.addEventListener("resize", scheduleResize);
    }

    // First plot must run before graphEl.on (Plotly attaches .on only after plot).
    await loadCatalog();

    graphEl.on("plotly_relayout", onRelayout);
    graphEl.on("plotly_hover", (ev) => {
      if (!ev?.points?.length) return;
      const pt = ev.points[0];
      if (pt.data?.name === "__anomaly_markers") return;
      // In 값 탐색, the locked cursor drives the panel until ←/→ or click.
      if (interactionMode === "inspect" && inspectIndex != null) return;
      const ms = parseTime(pt.x);
      const idx = nearestTimeIndex(ms);
      if (idx == null) return;
      refreshHoverPanel(idx);
    });
    graphEl.on("plotly_click", (ev) => {
      if (!payload || !ev?.points?.length) return;
      const pt = ev.points[0];
      if (interactionMode === "inspect") {
        if (pt.data?.name === "__anomaly_markers") return;
        const ms = parseTime(pt.x);
        if (isFinite(ms)) selectInspectAtMs(ms);
        return;
      }
      if (!anomaliesEnabled()) return;
      const id = pt.customdata;
      if (id == null || pt.data?.name !== "__anomaly_markers") return;
      if (Date.now() < _tapLockUntil) return;
      _tapLockUntil = Date.now() + 350;
      toggleLabelFromGraph(id);
    });
    scheduleResize();
  }

  init().catch((err) => {
    console.error(err);
    setStatus(String(err.message || err));
  });
})();
