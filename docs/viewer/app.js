(() => {
  const DATA_BASE = "data";
  const graphEl = document.getElementById("graph");
  const selectEl = document.getElementById("plmn-select");
  const listEl = document.getElementById("label-list");
  const statusEl = document.getElementById("status");

  let catalog = [];
  let payload = null;
  let selectedId = null;
  let dragmode = "pan";
  let xRange = null;
  let yRange = null;
  let yAuto = true;
  let suppressRelayout = false;

  function setStatus(text) {
    statusEl.textContent = text || "";
  }

  function syncYAutoButton() {
    const btn = document.getElementById("btn-y-auto");
    if (btn) btn.classList.toggle("active", yAuto);
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

  /** Y auto from values visible in [x0ms, x1ms] (no forced zero). */
  function yRangeForWindow(data, x0ms, x1ms) {
    if (!data?.t?.length) return null;
    const metrics = Object.values(data.metrics || {});
    if (!metrics.length) return null;

    let lo = 0;
    let hi = data.t.length - 1;
    if (isFinite(x0ms) && isFinite(x1ms) && x1ms > x0ms) {
      // Bound scan roughly by binary search on t.
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
    }

    let ymin = Infinity;
    let ymax = -Infinity;
    for (let i = lo; i <= hi; i++) {
      const ms = parseTime(data.t[i]);
      if (isFinite(x0ms) && ms < x0ms) continue;
      if (isFinite(x1ms) && ms > x1ms) continue;
      for (const series of metrics) {
        const v = series[i];
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

  /** Keep the current Y floor fixed; only the top moves (Y+ = 0.7, Y- = 1.4). */
  function scaleY(factor) {
    const bounds = currentYBounds();
    if (!bounds) return;
    let [lo, hi] = bounds;
    // Prefer a non-negative floor when the visible series is non-negative.
    const win = visibleXWindowMs();
    const dataYr = win
      ? yRangeForWindow(payload, win[0], win[1])
      : null;
    if (dataYr && dataYr[0] >= 0 && lo < 0) lo = 0;
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
    const traces = Object.entries(data.metrics || {}).map(([name, y]) => ({
      type: "scattergl",
      mode: "lines",
      name,
      x: data.t,
      y,
      opacity: 0.55,
      line: { width: 1 },
      hovertemplate:
        "<b>%{fullData.name}</b><br>%{x}<br>값=%{y}<extra></extra>",
    }));

    let x0ms = null;
    let x1ms = null;
    if (xRange) {
      x0ms = parseTime(xRange[0]);
      x1ms = parseTime(xRange[1]);
    } else if (data.t?.length) {
      x0ms = parseTime(data.t[0]);
      x1ms = parseTime(data.t[data.t.length - 1]);
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
      autorange: !xRange,
    };
    if (xRange) xaxis.range = xRange;

    const yaxis = {
      title: "value",
      fixedrange: true,
      autorange: !yr,
      rangemode: "normal",
    };
    if (yr) yaxis.range = yr;

    const layout = {
      margin: { l: 52, r: 20, t: 36, b: 48 },
      height: 420,
      autosize: true,
      showlegend: false,
      hovermode: "closest",
      dragmode,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#fffdf8",
      title: {
        text: `${data.display || data.plmn} · labels=${(data.labels || []).length}`,
        font: { size: 14 },
      },
      xaxis,
      yaxis,
      shapes: shapesForLabels(data.labels, highlightId),
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
    });
    suppressRelayout = false;
  }

  function applyXYRelayout(x0ms, x1ms) {
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
      const pad = Math.max((hi - lo) * 0.5, 6 * 3600 * 1000);
      if (hi === lo) {
        lo -= pad;
        hi += pad;
      } else {
        lo -= pad;
        hi += pad;
      }
      xRange = [msToPlotNaive(lo), msToPlotNaive(hi)];
      yAuto = true;
      syncYAutoButton();
      setStatus(`선택: ${item.line || item.id}`);
    } else if (item) {
      setStatus(`선택: ${item.line || item.id}`);
    }
    await draw();
  }

  function setDragMode(mode) {
    dragmode = mode;
    document.getElementById("btn-pan").classList.toggle("active", mode === "pan");
    document.getElementById("btn-zoom").classList.toggle("active", mode === "zoom");
    if (graphEl && graphEl.data) {
      Plotly.relayout(graphEl, { dragmode: mode });
    }
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
  }

  async function loadPlmn(plmn) {
    setStatus("로딩 중…");
    selectedId = null;
    xRange = null;
    yRange = null;
    yAuto = true;
    syncYAutoButton();
    const res = await fetch(`${DATA_BASE}/${plmn}.json`);
    if (!res.ok) throw new Error(`failed to load ${plmn}`);
    payload = await res.json();
    renderLabelList();
    await draw();
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
      draw();
      return;
    }

    const cur = currentXRangeMs();
    if (!cur || !(cur[1] > cur[0])) return;
    applyXYRelayout(cur[0], cur[1]);
  }

  async function init() {
    const res = await fetch(`${DATA_BASE}/index.json`);
    if (!res.ok) {
      setStatus("data/index.json 없음 — export_viewer.py 를 실행하세요.");
      return;
    }
    const idx = await res.json();
    // Public viewer: only operators that currently have labels.
    catalog = (idx.plmns || []).filter((row) => (row.n_labels || 0) > 0);
    selectEl.innerHTML = "";
    for (const row of catalog) {
      const opt = document.createElement("option");
      opt.value = row.plmn;
      opt.textContent = `#${String(row.rank ?? "").padStart(3, "0")} ${row.display} (${row.n_labels})`;
      selectEl.appendChild(opt);
    }
    if (!catalog.length) {
      setStatus("내보낼 사업자가 없습니다.");
      return;
    }
    selectEl.addEventListener("change", () => loadPlmn(selectEl.value));
    document.getElementById("btn-pan").onclick = () => setDragMode("pan");
    document.getElementById("btn-zoom").onclick = () => setDragMode("zoom");
    document.getElementById("btn-zoom-in").onclick = () => scaleX(0.7, false);
    document.getElementById("btn-zoom-out").onclick = () => scaleX(1 / 0.7, false);
    document.getElementById("btn-reset").onclick = () => resetX();
    document.getElementById("btn-y-in").onclick = () => scaleY(0.7);
    document.getElementById("btn-y-out").onclick = () => scaleY(1.4);
    document.getElementById("btn-y-auto").onclick = () => resetYAuto();
    syncYAutoButton();

    // First plot must run before graphEl.on (Plotly attaches .on only after plot).
    await loadPlmn(catalog[0].plmn);

    graphEl.on("plotly_relayout", onRelayout);
    graphEl.on("plotly_click", (ev) => {
      const x = ev?.points?.[0]?.x;
      if (x == null || !payload) return;
      const ts = parseTime(x);
      let best = null;
      let bestScore = Infinity;
      for (const item of payload.labels || []) {
        const a = parseTime(utcIsoToPlotNaive(item.start));
        const b = parseTime(utcIsoToPlotNaive(item.end || item.start));
        const kind = (item.kind || "point").toLowerCase();
        if (kind === "point" || item.start === item.end) {
          const d = Math.abs(ts - a);
          if (d < bestScore && d <= 10 * 60 * 1000) {
            bestScore = d;
            best = item;
          }
        } else if (ts >= Math.min(a, b) && ts <= Math.max(a, b)) {
          const span = Math.abs(b - a);
          if (span < bestScore) {
            bestScore = span;
            best = item;
          }
        }
      }
      if (best) selectLabel(best.id, false);
    });
  }

  init().catch((err) => {
    console.error(err);
    setStatus(String(err.message || err));
  });
})();
