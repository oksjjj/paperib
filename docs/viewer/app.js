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
  /** Show/hide anomaly overlays on the current chart (does not filter PLMNs). */
  let showAnomalies = true;
  let _resizeTimer = null;
  const coarsePointer =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;

  function setStatus(text) {
    statusEl.textContent = text || "";
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
    for (const series of Object.values(payload.metrics || {})) {
      const v = series[best];
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

    if (showAnomalies) {
      const markers = anomalyMarkerTrace(data.labels, highlightId);
      if (markers) traces.push(markers);
    }

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
      height: graphHeightPx(),
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
      shapes: showAnomalies
        ? shapesForLabels(data.labels, highlightId)
        : [],
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
    if (xRange) {
      post["xaxis.range"] = xRange;
      post["xaxis.autorange"] = false;
    }
    await Plotly.relayout(graphEl, post);
    suppressRelayout = false;
    bindAnomalyShapeClicks();
    scheduleResize();
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
      // Tight fit, then zoom-out ×7 (same idea as labeling "선택 라벨로 줌").
      const tight = Math.max(hi - lo, 30 * 60 * 1000);
      const mid = (lo + hi) / 2;
      const half = (tight * Math.pow(1 / 0.7, 7)) / 2;
      lo = mid - half;
      hi = mid + half;
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
    const res = await fetch(`${DATA_BASE}/${plmn}.json`, { cache: "no-store" });
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
    document.getElementById("btn-pan").onclick = () => setDragMode("pan");
    document.getElementById("btn-zoom").onclick = () => setDragMode("zoom");
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
    syncYAutoButton();

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
    graphEl.on("plotly_click", (ev) => {
      if (!showAnomalies || !payload || !ev?.points?.length) return;
      const pt = ev.points[0];
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
