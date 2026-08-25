(() => {
  const DATA_BASE = "data";
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
  const graphEl = document.getElementById("graph");
  const selectEl = document.getElementById("plmn-select");
  const listEl = document.getElementById("label-list");
  const statusEl = document.getElementById("status");
  const metricListEl = document.getElementById("metric-filter-list");
  const hoverPanelEl = document.getElementById("hover-panel");

  let catalog = [];
  let payload = null;
  let selectedId = null;
  /** pan | zoom | inspect */
  let interactionMode = "pan";
  let dragmode = "pan";
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

  /** Metrics ranked by sum over the current on-screen time window (desc). */
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
      for (let i = lo0; i <= hi0; i++) {
        const ms = parseTime(payload.t[i]);
        if (ms < x0ms || ms > x1ms) continue;
        const v = series[i];
        if (v != null && isFinite(v)) sum += v;
      }
      return { name, sum };
    });
    scored.sort((a, b) => b.sum - a.sum || a.name.localeCompare(b.name));
    return scored.map((x) => x.name);
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
      lab.appendChild(document.createTextNode(name));
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
    const total = allMetricNames().length;
    setStatus(
      keepY && n === 0
        ? "metric 전체 해제 (Y축 유지)"
        : `표시 metric ${n}/${total}`
    );
    renderMetricFilter();
    refreshHoverPanel();
    await draw();
  }

  function selectAllMetrics() {
    visibleMetrics = new Set(allMetricNames());
    onVisibleMetricsChanged({ keepY: false });
  }

  function clearAllMetrics() {
    visibleMetrics = new Set();
    onVisibleMetricsChanged({ keepY: true });
  }

  function formatMetricValue(v) {
    if (v == null || !isFinite(v)) return "—";
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString("en-US");
    if (Number.isInteger(v)) return String(v);
    return String(Math.round(v * 1000) / 1000);
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
        return { name, val: v == null || !isFinite(v) ? -Infinity : Number(v) };
      })
      .sort((a, b) => b.val - a.val);
    const when = payload.t[index] || "";
    const cells = pairs
      .map((p, i) => {
        const bg = Math.floor(i / 4) % 2 ? "#f6f8fa" : "#ffffff";
        const shown = isFinite(p.val) ? formatMetricValue(p.val) : "—";
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
      const lab = nLab > 0 ? ` · labels ${nLab}` : "";
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
    for (const item of labels || []) {
      const kind = (item.kind || "point").toLowerCase();
      const x0 = utcIsoToPlotNaive(item.start);
      const x1 = utcIsoToPlotNaive(item.end || item.start);
      const hi = item.id === highlightId;
      const stroke = hi ? "#c9a227" : "crimson";
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
          fillcolor: hi ? "rgba(201,162,39,0.28)" : "rgba(220,20,60,0.28)",
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
      if (!visibleMetrics.has(name)) continue;
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
    for (const item of labels || []) {
      const kind = (item.kind || "point").toLowerCase();
      const edges =
        kind === "point" || item.start === item.end
          ? [utcIsoToPlotNaive(item.start)]
          : [
              utcIsoToPlotNaive(item.start),
              utcIsoToPlotNaive(item.end || item.start),
            ];
      const hi = item.id === highlightId;
      for (const x of edges) {
        const y = nearestEnvelopeY(parseTime(x));
        if (y == null) continue;
        xs.push(x);
        ys.push(y);
        colors.push(hi ? "#c9a227" : "crimson");
        sizes.push(hi ? baseSize + 4 : baseSize);
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
        const v = data.metrics[name]?.[i];
        if (v == null || !isFinite(v)) continue;
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
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
    const traces = colorOrder
      .filter((name) => visibleMetrics.has(name))
      .map((name) => {
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
            "값=%{y}<extra></extra>",
          uid: name,
        };
      });

    if (showAnomalies) {
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
      margin: { l: 52, r: 20, t: 36, b: 48 },
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
        text: `${data.display || data.plmn} · labels=${(data.labels || []).length}`,
        font: { size: 14 },
      },
      xaxis,
      yaxis,
      shapes: (() => {
        const out = showAnomalies
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
          });
        }
        return out;
      })(),
    };

    return { data: traces, layout };
  }

  async function draw() {
    if (!payload) return;
    suppressRelayout = true;
    const fig = buildFigure(payload, selectedId);
    await Plotly.react(graphEl, fig.data, fig.layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      // Help mobile: scroll parent, pan plot
      scrollZoom: false,
    });
    // scattergl often keeps a stale viewport until an explicit resize + Y pin.
    const win = visibleXWindowMs();
    let yr = yRange;
    if (yAuto && win) yr = syncYFromX(win[0], win[1]);
    const post = { height: graphHeightPx() };
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
    suppressRelayout = false;
    bindAnomalyShapeClicks();
    bindInspectClicks();
    installPlaceTimeTip();
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
      if (!showAnomalies || !payload) return false;
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
      selectLabel(hit.id, true);
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

  function applyXYRelayout(x0ms, x1ms) {
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
    Plotly.relayout(graphEl, update).then(() => {
      suppressRelayout = false;
      renderMetricFilter();
      refreshHoverPanel();
    });
  }

  function renderLabelList() {
    listEl.innerHTML = "";
    const labels = payload?.labels || [];
    if (!labels.length) {
      listEl.innerHTML = '<p class="hint">라벨이 없습니다.</p>';
      return;
    }
    for (const item of labels) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "label-item" + (item.id === selectedId ? " selected" : "");
      btn.textContent = item.line || `${item.kind} ${item.id}`;
      btn.addEventListener("click", () => selectLabel(item.id, true));
      listEl.appendChild(btn);
    }
  }

  async function selectLabel(id, zoomTo) {
    selectedId = id;
    const item = (payload.labels || []).find((x) => x.id === id);
    renderLabelList();
    if (item && zoomTo) {
      const a = parseTime(utcIsoToPlotNaive(item.start));
      const b = parseTime(utcIsoToPlotNaive(item.end || item.start));
      let lo = Math.min(a, b);
      let hi = Math.max(a, b);
      // Tight fit, then zoom-out ×7 (same idea as labeling "선택 라벨로 줌").
      const tight = Math.max(hi - lo, 30 * 60 * 1000);
      const mid = (lo + hi) / 2;
      const half = (tight * Math.pow(1 / 0.7, 7)) / 2;
      lo = mid - half;
      hi = mid + half;
      [lo, hi] = clampTimeWindow(lo, hi);
      xRange = [msToPlotNaive(lo), msToPlotNaive(hi)];
      yAuto = true;
      syncYAutoButton();
      // Compute Y before draw so both react + post-relayout share the same range.
      const yr = syncYFromX(lo, hi);
      setStatus(`선택: ${item.line || item.id}`);
      await draw();
      if (yr) {
        suppressRelayout = true;
        try {
          await Plotly.relayout(graphEl, {
            "xaxis.range": xRange,
            "xaxis.autorange": false,
            "yaxis.range": yr,
            "yaxis.autorange": false,
            height: graphHeightPx(),
          });
        } finally {
          suppressRelayout = false;
        }
        scheduleResize();
      }
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
    if (showAnomalies === on) return;
    showAnomalies = on;
    syncOverlayButtons();
    await draw();
    setStatus(on ? "anomaly 표시" : "anomaly 숨김");
  }

  function setInteractionMode(mode) {
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
    draw().then(() => {
      refreshHoverPanel(
        interactionMode === "inspect" ? inspectIndex : hoverIndex
      );
      if (mode === "inspect") {
        setStatus("값 탐색: 클릭 또는 ←/→ · 드래그 이동 없음");
      }
    });
  }

  function selectInspectAtMs(ms) {
    if (!payload?.t?.length || !isFinite(ms)) return;
    const idx = nearestTimeIndex(ms);
    if (idx == null) return;
    inspectIndex = idx;
    hoverIndex = idx;
    refreshHoverPanel(idx);
    setStatus(`값 탐색: ${payload.t[idx]} · ←/→ 키로 이동`);
    draw();
  }

  function stepInspect(delta) {
    if (interactionMode !== "inspect" || !payload?.t?.length) return;
    let idx = inspectIndex;
    if (idx == null) idx = hoverIndex;
    if (idx == null) idx = defaultHoverIndex();
    if (idx == null) return;
    idx = Math.max(0, Math.min(payload.t.length - 1, idx + delta));
    inspectIndex = idx;
    hoverIndex = idx;
    refreshHoverPanel(idx);
    setStatus(`값 탐색: ${payload.t[idx]} · ←/→ 키로 이동`);
    draw();
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
    if (!layout?.xaxis?.range) return null;
    const r = layout.xaxis.range;
    return [parseTime(r[0]), parseTime(r[1])];
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
    applyXYRelayout(a, b);
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
    const res = await fetch(`${DATA_BASE}/${plmn}.json`, { cache: "no-store" });
    if (!res.ok) throw new Error(`failed to load ${plmn}`);
    payload = await res.json();
    visibleMetrics = new Set(allMetricNames());
    hoverIndex = defaultHoverIndex();
    if (interactionMode === "inspect") inspectIndex = hoverIndex;
    renderMetricFilter();
    renderLabelList();
    await draw();
    refreshHoverPanel(hoverIndex);
    setStatus(
      `${payload.start_kst} ~ ${payload.end_kst} · ${payload.n_points} pts · ${
        (payload.labels || []).length
      } labels`
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

    const cur = currentXRangeMs();
    if (!cur || !(cur[1] > cur[0])) return;
    // Reject bogus windows from empty-plot autorange (no overlap with data).
    if (payload.t?.length) {
      const d0 = parseTime(payload.t[0]);
      const d1 = parseTime(payload.t[payload.t.length - 1]);
      if (!(d1 > d0)) return;
      if (cur[1] < d0 || cur[0] > d1) return;
    }
    applyXYRelayout(cur[0], cur[1]);
  }

  async function init() {
    const res = await fetch(`${DATA_BASE}/index.json`, { cache: "no-store" });
    if (!res.ok) {
      setStatus("data/index.json 없음 — export_viewer.py 를 실행하세요.");
      return;
    }
    const idx = await res.json();
    catalog = idx.plmns || [];
    syncOverlayButtons();
    fillPlmnSelect(catalog);
    if (!catalog.length) {
      setStatus("내보낼 사업자가 없습니다.");
      return;
    }
    selectEl.addEventListener("change", () => loadPlmn(selectEl.value));
    document.getElementById("btn-pan").onclick = () =>
      setInteractionMode("pan");
    document.getElementById("btn-zoom").onclick = () =>
      setInteractionMode("zoom");
    document.getElementById("btn-inspect").onclick = () =>
      setInteractionMode("inspect");
    document.getElementById("btn-zoom-in").onclick = () => scaleX(0.7, false);
    document.getElementById("btn-zoom-out").onclick = () => scaleX(1 / 0.7, false);
    document.getElementById("btn-reset").onclick = () => resetX();
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
    await loadPlmn(catalog[0].plmn);

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
      if (!showAnomalies) return;
      const id = pt.customdata;
      if (id == null || pt.data?.name !== "__anomaly_markers") return;
      if (Date.now() < _tapLockUntil) return;
      _tapLockUntil = Date.now() + 350;
      selectLabel(id, true);
    });
    scheduleResize();
  }

  init().catch((err) => {
    console.error(err);
    setStatus(String(err.message || err));
  });
})();
