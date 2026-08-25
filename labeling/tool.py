"""Anomaly labeling helpers: ranking, data load, label store, plotting."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

KST = ZoneInfo("Asia/Seoul")
# Display: Korean date style in Asia/Seoul (e.g. 2026년 04월 30일 00:00)
KST_FMT = "%Y년 %m월 %d일 %H:%M"
KST_FMT_SEC = "%Y년 %m월 %d일 %H:%M:%S"
_KST_PARSE_FMTS = (
    "%Y년 %m월 %d일 %H:%M:%S",
    "%Y년 %m월 %d일 %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ANALYSIS_RANK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "analysis", "plmn_m971_sum_sorted.csv"
)
# Private artifacts stay under labeling/ (gitignored)
LABEL_DIR = os.path.join(os.path.dirname(__file__), "labels")
RANK_CACHE_PATH = os.path.join(LABEL_DIR, "plmn_rank.csv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RANK_SIGNATURE_PATH = os.path.join(CACHE_DIR, "plmn_rank.signature")

TAG_COLORS = {
    # Stronger fills so ranges stay visible on white plot background.
    "anomaly": "rgba(200, 16, 46, 0.38)",
    "normal": "rgba(20, 120, 50, 0.32)",
    "uncertain": "rgba(200, 110, 0, 0.36)",
}
TAG_LINE = {
    "anomaly": "#b01028",
    "normal": "#147832",
    "uncertain": "#c86e00",
}

# Mid-tone hues: readable on white without the previous near-black density.
SERIES_COLORWAY = (
    "#3d7ab5",  # blue
    "#d9655a",  # coral red
    "#3da86a",  # green
    "#9b6bb8",  # purple
    "#d4a017",  # gold
    "#4a90a4",  # steel
    "#c06a5a",  # brick
    "#2e9a85",  # teal
    "#8e6ba8",  # soft purple
    "#d4833a",  # orange
    "#5b8fbf",  # sky
    "#c97b72",  # rose
    "#4caf77",  # mint green
    "#a07cbc",  # lilac
    "#b8a03a",  # olive gold
    "#5b9bd5",  # light steel
    "#b8875a",  # tan
    "#3cb09a",  # sea green
    "#9a7aab",  # plum
    "#c97a55",  # rust
)

# Shapes/annotations tagged with these names are transient labeling guides.
PENDING_ANCHOR_NAME = "pending_range_anchor"
PENDING_RANGE_FILL_NAME = "pending_range_fill"
LABEL_HIGHLIGHT_NAME = "label_highlight"
LABEL_HIGHLIGHT_START = "label_highlight_start"
LABEL_HIGHLIGHT_END = "label_highlight_end"
VALUE_CURSOR_NAME = "value_cursor"


MAPPING_DIR = os.path.join(os.path.dirname(__file__), "..", "mapping")
PLMN_MAPPING_PATH = os.path.join(MAPPING_DIR, "plmn_mapping.txt")
METRIC_MAPPING_PATH = os.path.join(MAPPING_DIR, "metric_mapping.txt")

_PLMN_ORIGINAL: dict[str, str] | None = None
_METRIC_ORIGINAL: dict[str, str] | None = None
_MAPPING_ENABLED = True


def set_mapping_enabled(enabled: bool) -> None:
    """Enable/disable mapping lookups (PLMN/metric original names).

    When disabled, display helpers keep masked IDs only — safe for shared/git
    notebooks even if mapping/ exists locally.
    """
    global _MAPPING_ENABLED
    _MAPPING_ENABLED = bool(enabled)


def mapping_enabled() -> bool:
    return _MAPPING_ENABLED


def _load_mapping_tables() -> None:
    """Lazy-load mapping files if present under mapping/."""
    global _PLMN_ORIGINAL, _METRIC_ORIGINAL
    if _PLMN_ORIGINAL is not None and _METRIC_ORIGINAL is not None:
        return

    _PLMN_ORIGINAL = {}
    _METRIC_ORIGINAL = {}

    if os.path.exists(PLMN_MAPPING_PATH):
        plmn_df = pd.read_csv(PLMN_MAPPING_PATH, sep="\t")
        _PLMN_ORIGINAL = {
            str(row["masked_plmn"]): str(row["original_plmn"])
            for _, row in plmn_df.iterrows()
        }

    if os.path.exists(METRIC_MAPPING_PATH):
        metric_df = pd.read_csv(METRIC_MAPPING_PATH, sep="\t")
        _METRIC_ORIGINAL = {
            str(row["masked_metric"]): str(row["original_metric"])
            for _, row in metric_df.iterrows()
        }


def reload_mappings() -> None:
    """Force re-read mapping files (e.g. after replacing mapping/)."""
    global _PLMN_ORIGINAL, _METRIC_ORIGINAL
    _PLMN_ORIGINAL = None
    _METRIC_ORIGINAL = None
    _load_mapping_tables()


def plmn_original(plmn: str) -> str | None:
    if not _MAPPING_ENABLED:
        return None
    _load_mapping_tables()
    assert _PLMN_ORIGINAL is not None
    return _PLMN_ORIGINAL.get(plmn)


def metric_original(metric: str) -> str | None:
    if not _MAPPING_ENABLED:
        return None
    _load_mapping_tables()
    assert _METRIC_ORIGINAL is not None
    return _METRIC_ORIGINAL.get(metric)


def plmn_original_short(plmn: str) -> str | None:
    """Original PLMN without 'PLMN_' prefix, e.g. PLMN_46011 -> 46011."""
    original = plmn_original(plmn)
    if original is None:
        return None
    if original.startswith("PLMN_"):
        return original[len("PLMN_") :]
    return original


def display_plmn(plmn: str) -> str:
    """P0480 or P0480 (46011) when mapping exists."""
    short = plmn_original_short(plmn)
    return f"{plmn} ({short})" if short else plmn


def display_metric(metric: str) -> str:
    """M971 or M971 (COMB_ATTEMPT) when mapping exists."""
    original = metric_original(metric)
    return f"{metric} ({original})" if original else metric


def to_kst(ts) -> pd.Timestamp:
    """Normalize any timestamp to Asia/Seoul."""
    t = pd.to_datetime(ts, utc=True)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(KST)


def format_kst(ts, *, seconds: bool = False) -> str:
    """Korean date format in KST, e.g. 2026년 04월 30일 00:00."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return ""
    t = to_kst(ts)
    return t.strftime(KST_FMT_SEC if seconds else KST_FMT)


def to_plot_time(ts) -> pd.Timestamp:
    """KST wall-clock as naive Timestamp for Plotly axis display."""
    return to_kst(ts).tz_localize(None)


def to_plot_times(values) -> np.ndarray:
    """Vectorized KST-naive datetimes for Plotly x values."""
    s = pd.to_datetime(pd.Series(values), utc=True)
    return s.dt.tz_convert(KST).dt.tz_localize(None).to_numpy()


def parse_time(value) -> pd.Timestamp:
    """Parse user/UI time to UTC. Naive strings are treated as KST."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        raise ValueError("empty timestamp")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        s = str(value).strip()
        ts = None
        if "년" in s:
            for fmt in _KST_PARSE_FMTS:
                try:
                    ts = pd.Timestamp(datetime.strptime(s, fmt))
                    break
                except ValueError:
                    continue
        if ts is None:
            ts = pd.to_datetime(s, utc=False)
    if getattr(ts, "tzinfo", None) is None or ts.tzinfo is None:
        ts = ts.tz_localize(KST)
    return ts.tz_convert("UTC")


def ensure_label_dir() -> str:
    os.makedirs(LABEL_DIR, exist_ok=True)
    return LABEL_DIR


def load_or_build_ranking(top_n: int | None = None) -> pd.DataFrame:
    """Load a ranking that matches the current source-data snapshot.

    Adding or replacing data/*.csv invalidates the rank signature. The next app
    start rebuilds the ranking, while label JSON files remain untouched.
    """
    ensure_label_dir()
    os.makedirs(CACHE_DIR, exist_ok=True)
    signature = _data_signature(DATA_DIR)
    cached_signature = None
    if os.path.exists(RANK_SIGNATURE_PATH):
        try:
            with open(RANK_SIGNATURE_PATH, encoding="utf-8") as f:
                cached_signature = f.read().strip()
        except OSError:
            pass

    if os.path.exists(RANK_CACHE_PATH) and cached_signature == signature:
        rank_df = pd.read_csv(RANK_CACHE_PATH)
    elif any(f.endswith(".csv") for f in os.listdir(DATA_DIR)):
        rank_df = _build_ranking_from_data()
        save_ranking_cache(rank_df, signature)
    elif os.path.exists(RANK_CACHE_PATH):
        rank_df = pd.read_csv(RANK_CACHE_PATH)
    elif os.path.exists(ANALYSIS_RANK_PATH):
        rank_df = pd.read_csv(ANALYSIS_RANK_PATH)
    else:
        raise FileNotFoundError(f"No source CSV or ranking cache found under {DATA_DIR}")

    if "rank" not in rank_df.columns:
        rank_df = rank_df.sort_values("M971_sum", ascending=False).reset_index(drop=True)
        rank_df.insert(0, "rank", rank_df.index + 1)

    if top_n is not None:
        rank_df = rank_df.head(top_n).copy()

    rank_df = rank_df.reset_index(drop=True)
    if "PLMN" in rank_df.columns:
        rank_df["PLMN_name"] = [plmn_original_short(p) or "" for p in rank_df["PLMN"]]
        rank_df["PLMN_display"] = [display_plmn(p) for p in rank_df["PLMN"]]
    return rank_df


def save_ranking_cache(rank_df: pd.DataFrame, signature: str | None = None) -> None:
    """Persist ranking and the source-data signature used to create it."""
    ensure_label_dir()
    os.makedirs(CACHE_DIR, exist_ok=True)
    rank_df.to_csv(RANK_CACHE_PATH, index=False)
    with open(RANK_SIGNATURE_PATH, "w", encoding="utf-8") as f:
        f.write(signature or _data_signature(DATA_DIR))


def _build_ranking_from_data() -> pd.DataFrame:
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    sums: dict[str, int] = {}
    for file in csv_files:
        chunk = pd.read_csv(os.path.join(DATA_DIR, file), usecols=["PLMN", "M971"])
        part = chunk.groupby("PLMN")["M971"].sum()
        for plmn, value in part.items():
            sums[plmn] = sums.get(plmn, 0) + int(value)
    rank_df = (
        pd.DataFrame({"PLMN": list(sums.keys()), "M971_sum": list(sums.values())})
        .sort_values("M971_sum", ascending=False)
        .reset_index(drop=True)
    )
    rank_df.insert(0, "rank", rank_df.index + 1)
    return rank_df


def _data_signature(data_dir: str) -> str:
    """Fingerprint of the data dir so caches invalidate when files change.

    Keep this format stable: changing it invalidates every cached pickle and
    forces a full CSV rescan per PLMN on the next load.
    """
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    newest = 0.0
    total = 0
    for file in csv_files:
        stat = os.stat(os.path.join(data_dir, file))
        newest = max(newest, stat.st_mtime)
        total += stat.st_size
    return f"{len(csv_files)}-{int(newest)}-{total}"


def load_plmn(
    plmn: str,
    data_dir: str = DATA_DIR,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load one PLMN's series. Caches to cache/ so repeat loads are fast."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{plmn}.pkl")
    signature = _data_signature(data_dir)

    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
            if cached.attrs.get("signature") == signature:
                return cached
        except Exception:
            pass

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    parts = []
    for file in csv_files:
        temp = pd.read_csv(os.path.join(data_dir, file))
        temp = temp[temp["PLMN"] == plmn]
        if not temp.empty:
            parts.append(temp)
    if not parts:
        raise ValueError(f"No rows found for PLMN={plmn}")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.attrs["signature"] = signature

    if use_cache:
        try:
            df.to_pickle(cache_path)
        except Exception:
            pass
    return df


def clear_plmn_cache(plmn: str | None = None) -> int:
    """Delete cached PLMN frames. Returns number of files removed."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    targets = (
        [f"{plmn}.pkl"]
        if plmn
        else [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    )
    removed = 0
    for name in targets:
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
    return removed


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("time", "PLMN")]


def label_path(plmn: str) -> str:
    ensure_label_dir()
    return os.path.join(LABEL_DIR, f"{plmn}_labels.json")


def empty_label_doc(plmn: str, rank: int | None = None) -> dict[str, Any]:
    return {
        "plmn": plmn,
        "rank": rank,
        "updated_at": None,
        "labels": [],
    }


def load_labels(plmn: str, rank: int | None = None) -> dict[str, Any]:
    path = label_path(plmn)
    if not os.path.exists(path):
        return empty_label_doc(plmn, rank)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc.setdefault("plmn", plmn)
    if rank is not None:
        doc["rank"] = rank
    doc.setdefault("labels", [])
    return doc


def save_labels(doc: dict[str, Any]) -> str:
    doc = dict(doc)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = label_path(doc["plmn"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def labels_to_frame(doc: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in doc.get("labels", []):
        metrics = item.get("metrics") or ["ALL"]
        kind = item.get("kind", "point")
        start = format_kst(item.get("start"))
        end = format_kst(item.get("end"))
        interval = start if kind == "point" else f"{start} → {end}"
        rows.append(
            {
                "id": item.get("id"),
                "형식": "점" if kind == "point" else "구간",
                "tag": item.get("tag"),
                "시각 / 구간 (KST)": interval,
                "metrics": ",".join(
                    m if m == "ALL" else display_metric(m) for m in metrics
                ),
                "note": item.get("note", ""),
                "updated_at": format_kst(item.get("updated_at"), seconds=True),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "형식",
                "tag",
                "시각 / 구간 (KST)",
                "metrics",
                "note",
                "updated_at",
            ]
        )
    return pd.DataFrame(rows)


def add_label(
    doc: dict[str, Any],
    *,
    kind: str,
    tag: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    metrics: list[str] | None = None,
    note: str = "",
    label_id: str | None = None,
) -> dict[str, Any]:
    kind = kind.lower()
    tag = tag.lower()
    if kind not in {"point", "range"}:
        raise ValueError("kind must be 'point' or 'range'")
    if tag not in TAG_COLORS:
        raise ValueError(f"tag must be one of {list(TAG_COLORS)}")

    start_ts = parse_time(start)
    end_ts = start_ts if kind == "point" else parse_time(end)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    item = {
        "id": label_id or str(uuid.uuid4())[:8],
        "kind": kind,
        "tag": tag,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "metrics": metrics or ["ALL"],
        "note": note,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    labels = [x for x in doc.get("labels", []) if x.get("id") != item["id"]]
    labels.append(item)
    labels.sort(key=lambda x: x["start"])
    doc["labels"] = labels
    return doc


def remove_label(doc: dict[str, Any], label_id: str) -> dict[str, Any]:
    doc["labels"] = [x for x in doc.get("labels", []) if x.get("id") != label_id]
    return doc


def update_label(doc: dict[str, Any], label_id: str, **fields: Any) -> dict[str, Any]:
    found = None
    for item in doc.get("labels", []):
        if item.get("id") == label_id:
            found = item
            break
    if found is None:
        raise KeyError(f"label id not found: {label_id}")

    if "kind" in fields:
        found["kind"] = fields["kind"]
    if "tag" in fields:
        found["tag"] = fields["tag"]
    if "note" in fields:
        found["note"] = fields["note"]
    if "metrics" in fields:
        found["metrics"] = fields["metrics"]
    if "start" in fields:
        found["start"] = parse_time(fields["start"]).isoformat()
    if "end" in fields:
        found["end"] = parse_time(fields["end"]).isoformat()
    if found["kind"] == "point":
        found["end"] = found["start"]
    else:
        start_ts = parse_time(found["start"])
        end_ts = parse_time(found["end"])
        if end_ts < start_ts:
            found["start"], found["end"] = end_ts.isoformat(), start_ts.isoformat()
    found["updated_at"] = datetime.now(timezone.utc).isoformat()

    doc["labels"] = sorted(doc["labels"], key=lambda x: x["start"])
    return doc


def ranked_metric_pairs(
    row: pd.Series, cols: list[str], *, nonzero_only: bool = False
) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for col in cols:
        val = row[col]
        if pd.isna(val):
            continue
        fval = float(val)
        if nonzero_only and fval == 0:
            continue
        pairs.append((col, fval))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def format_metric_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.4g}"


# Highlighted in the "이 시점 특성값" panel (rank, name, and value).
HOVER_RED_METRICS = frozenset({"M855", "M037", "M430", "M162", "M618", "M520"})
HOVER_ORANGE_METRICS = frozenset({"M874", "M185", "M965", "M843", "M419"})


def ranked_hover_text(
    row: pd.Series,
    cols: list[str],
    *,
    sep: str = "<br>",
    top_n: int | None = 12,
) -> str:
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    total = len(pairs)
    shown = pairs if top_n is None else pairs[:top_n]
    lines = [f"{display_metric(name)}={format_metric_value(val)}" for name, val in shown]
    if top_n is not None and total > top_n:
        lines.append(f"... +{total - top_n} more (아래 패널 스크롤)")
    return sep.join(lines)


def ranked_hover_html(ts, row: pd.Series, cols: list[str], *, cols_per_row: int = 4) -> str:
    """Scrollable grid of metric values, largest first, left-to-right then wrap."""
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    items = []
    for i, (name, val) in enumerate(pairs, start=1):
        bg = "#f6f8fa" if ((i - 1) // cols_per_row) % 2 else "#ffffff"
        if name in HOVER_RED_METRICS and val != 0:
            color, weight = "crimson", "700"
        elif name in HOVER_ORANGE_METRICS and val != 0:
            color, weight = "darkorange", "700"
        else:
            color, weight = "inherit", "400"
        items.append(
            "<div style='display:flex;justify-content:space-between;gap:6px;"
            "padding:3px 6px;background:{bg};min-width:0;"
            "color:{color};font-weight:{weight}'>"
            "<span style='min-width:22px;flex:0 0 auto;"
            "color:{rank_color}'>{i}</span>"
            "<span style='flex:1 1 auto;white-space:nowrap;overflow:hidden;"
            "text-overflow:ellipsis' title='{label}'>{label}</span>"
            "<span style='font-variant-numeric:tabular-nums;flex:0 0 auto'>"
            "{val}</span>"
            "</div>".format(
                bg=bg,
                color=color,
                weight=weight,
                rank_color="#888" if color == "inherit" else color,
                i=i,
                label=display_metric(name),
                val=format_metric_value(val),
            )
        )
    return (
        "<div style='font-family:sans-serif;font-size:12px;width:100%'>"
        f"<div style='font-weight:600;margin-bottom:4px'>{format_kst(ts)}"
        f"<span style='color:#666;font-weight:400'> (KST) · {len(pairs)}개 특성 · 값 내림차순</span>"
        "</div>"
        f"<div style='display:grid;grid-template-columns:repeat({cols_per_row},minmax(0,1fr));"
        f"gap:2px 10px'>" + "".join(items) + "</div>"
        "</div>"
    )


def data_time_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) of the loaded time series (data only, not wall-clock now)."""
    tmin = pd.to_datetime(df["time"].min(), utc=True)
    tmax = pd.to_datetime(df["time"].max(), utc=True)
    return tmin, tmax


def clamp_time_range(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    tmin: pd.Timestamp,
    tmax: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Keep [start, end] inside [tmin, tmax], preserving window width when possible."""
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    width = end_ts - start_ts
    full = tmax - tmin
    if width >= full:
        return tmin, tmax
    if start_ts < tmin:
        start_ts = tmin
        end_ts = tmin + width
    if end_ts > tmax:
        end_ts = tmax
        start_ts = tmax - width
    if start_ts < tmin:
        start_ts = tmin
    return start_ts, end_ts


def _filter_df(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out["time"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        out = out[out["time"] <= pd.to_datetime(end, utc=True)]
    return out


def minmax_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    """Bucketed min/max envelope indices: keeps spikes while cutting point count."""
    n = int(values.shape[0])
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    buckets = max(1, max_points // 2)
    step = int(np.ceil(n / buckets))
    if step <= 1:
        return np.arange(n)

    v = np.asarray(values, dtype="float64")
    pad = buckets * step - n
    if pad > 0:
        v = np.concatenate([v, np.full(pad, np.nan)])
    grid = v.reshape(buckets, step)
    filled = ~np.all(np.isnan(grid), axis=1)
    base = (np.arange(buckets) * step)[filled]
    sub = grid[filled]
    imax = np.argmax(np.where(np.isnan(sub), -np.inf, sub), axis=1)
    imin = np.argmin(np.where(np.isnan(sub), np.inf, sub), axis=1)
    idx = np.concatenate([base + imax, base + imin, [0, n - 1]])
    idx = np.unique(idx)
    return idx[idx < n]


def plot_series(
    view: pd.DataFrame,
    cols: list[str],
    max_points: int = 0,
) -> dict[str, tuple[Any, Any]]:
    """Per-metric (x, y) arrays reduced to at most `max_points` samples each."""
    times = to_plot_times(view["time"])
    out: dict[str, tuple[Any, Any]] = {}
    for col in cols:
        values = view[col].to_numpy()
        idx = minmax_indices(values, max_points)
        out[col] = (times[idx], values[idx])
    return out


def window_positions(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> np.ndarray | None:
    """Row positions inside [start, end], or None when the window is everything."""
    n = len(df)
    if n == 0:
        return None
    mask = np.ones(n, dtype=bool)
    if start is not None:
        mask &= (df["time"] >= pd.to_datetime(start, utc=True)).to_numpy()
    if end is not None:
        mask &= (df["time"] <= pd.to_datetime(end, utc=True)).to_numpy()
    pos = np.flatnonzero(mask)
    return pos if 0 < len(pos) < n else None


def detail_window(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    pad_ratio: float = 0.35,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Widen the high-res window so pan/zoom edges stay detailed if ranges drift."""
    if start is None or end is None or not len(df):
        return start, end
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if end_ts <= start_ts:
        return start_ts, end_ts
    pad = (end_ts - start_ts) * pad_ratio
    tmin = pd.to_datetime(df["time"].min(), utc=True)
    tmax = pd.to_datetime(df["time"].max(), utc=True)
    return max(tmin, start_ts - pad), min(tmax, end_ts + pad)


def window_indices(
    values: np.ndarray,
    pos: np.ndarray | None,
    max_points: int,
    context_ratio: float = 0.3,
) -> np.ndarray:
    """Detail inside `pos`, optionally with a coarse envelope outside.

    Keeping coarse context means the trace still spans the whole dataset, so
    plotly's double-click autorange resets to the full time span.
    """
    if pos is None:
        return minmax_indices(values, max_points)
    ctx_points = max(200, int(max_points * context_ratio))
    return np.union1d(
        minmax_indices(values, ctx_points),
        pos[minmax_indices(values[pos], max_points)],
    )


def plot_series_window(
    df: pd.DataFrame,
    cols: list[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_points: int = 1500,
    *,
    include_context: bool = False,
    pad_ratio: float = 0.35,
) -> dict[str, tuple[Any, Any]]:
    """Per-metric (x, y) arrays at high resolution around the visible window.

    By default only the (padded) window is drawn at high res. The first/last
    samples are always kept so the trace still spans the dataset (helps Plotly
    keep a stable full time domain without coarse mid-context).
    """
    if not len(df):
        return {col: ([], []) for col in cols}
    detail_start, detail_end = detail_window(df, start, end, pad_ratio=pad_ratio)
    pos = window_positions(df, detail_start, detail_end)
    times = to_plot_times(df["time"])
    n = len(df)
    ends = np.array([0, n - 1], dtype=int) if n else np.array([], dtype=int)
    out: dict[str, tuple[Any, Any]] = {}
    for col in cols:
        values = df[col].to_numpy()
        if include_context:
            idx = window_indices(values, pos, max_points)
        elif pos is None:
            idx = minmax_indices(values, max_points)
        else:
            idx = np.union1d(ends, pos[minmax_indices(values[pos], max_points)])
        out[col] = (times[idx], values[idx])
    return out


def add_label_indicator(fig: go.Figure, item: dict[str, Any]) -> bool:
    """Draw a label overlay; return True for a range label."""
    s_utc = pd.to_datetime(item["start"], utc=True)
    e_utc = pd.to_datetime(item["end"], utc=True)
    s = to_plot_time(s_utc)
    e = to_plot_time(e_utc)
    tag = item.get("tag", "anomaly")
    color = TAG_COLORS.get(tag, TAG_COLORS["anomaly"])
    line = TAG_LINE.get(tag, "crimson")
    is_range = item.get("kind") == "range" or s_utc != e_utc

    if is_range:
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=color,
            layer="below",
            line_width=0,
            editable=False,
        )
    else:
        fig.add_vline(
            x=s,
            line_width=2,
            line_dash="solid",
            line_color=line,
            editable=False,
        )
    return is_range


def freeze_shape_editing(fig: go.Figure) -> go.Figure:
    """Mark every existing shape non-editable so fills cannot be dragged whole."""
    shapes = fig.layout.shapes
    if not shapes:
        return fig
    locked = []
    for shape in shapes:
        data = shape.to_plotly_json() if hasattr(shape, "to_plotly_json") else dict(shape)
        data["editable"] = False
        locked.append(data)
    fig.layout.shapes = tuple(locked)
    return fig


def label_line(item: dict[str, Any]) -> str:
    """One-line description of a label for the saved-label list."""
    kind = (item.get("kind") or "point").lower()
    start = format_kst(item.get("start"))
    end = format_kst(item.get("end"))
    if kind == "point":
        text = f"[점] anomaly · {start}"
    else:
        text = f"[구간] anomaly · {start} → {end}"
    note = (item.get("note") or "").strip()
    if note:
        text += f" · {note}"
    return f"{text}  ({item.get('id')})"


def label_highlight_overlays(
    item: dict[str, Any],
    *,
    editable: bool = False,
    line_width: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Thin red edges for the selected label.

    Keep editable=False; the app installs a custom horizontal drag on these edges
    in 구간 편집 mode (Plotly native shape edit shows handles everywhere).
    """
    s = to_plot_time(pd.to_datetime(item["start"], utc=True))
    e = to_plot_time(pd.to_datetime(item["end"], utc=True))
    is_range = item.get("kind") == "range" or s != e

    def _edge(x, name: str) -> dict[str, Any]:
        return dict(
            type="line",
            xref="x",
            yref="paper",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(color="crimson", width=int(line_width)),
            layer="above",
            editable=False,
            name=name,
        )

    if is_range:
        return [_edge(s, LABEL_HIGHLIGHT_START), _edge(e, LABEL_HIGHLIGHT_END)], []
    return [_edge(s, LABEL_HIGHLIGHT_START)], []


def value_cursor_overlays(ts) -> tuple[dict[str, Any], dict[str, Any]]:
    """Vertical cursor and timestamp annotation for keyboard value inspection."""
    x = to_plot_time(ts)
    shape = dict(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="royalblue", width=2, dash="dot"),
        layer="above",
        name=VALUE_CURSOR_NAME,
    )
    note = dict(
        x=x,
        y=1,
        xref="x",
        yref="paper",
        text=f"값 탐색 · {format_kst(ts)}",
        showarrow=False,
        yshift=-8,
        font=dict(size=11, color="white"),
        bgcolor="royalblue",
        borderpad=3,
        name=VALUE_CURSOR_NAME,
    )
    return shape, note


def pending_anchor_shape(ts) -> dict[str, Any]:
    """Dotted guide marking a range label's first click (start / left)."""
    x = to_plot_time(ts)
    return dict(
        type="line",
        xref="x",
        yref="paper",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        line=dict(color="royalblue", width=3, dash="dot"),
        layer="above",
        editable=False,
        name=PENDING_ANCHOR_NAME,
    )


def pending_anchor_annotation(ts) -> dict[str, Any]:
    return dict(
        x=to_plot_time(ts),
        y=1.0,
        xref="x",
        yref="paper",
        text="① 시작(왼쪽) — 오른쪽으로 이동 후 끝점 클릭",
        showarrow=False,
        yshift=-8,
        font=dict(size=11, color="white"),
        bgcolor="royalblue",
        borderpad=3,
        name=PENDING_ANCHOR_NAME,
    )


def pending_range_fill_shape(start_ts, end_ts=None) -> dict[str, Any]:
    """Crimson band preview from start → end (transparent until end > start)."""
    x0 = to_plot_time(start_ts)
    x1 = to_plot_time(end_ts) if end_ts is not None else x0
    show = end_ts is not None and end_ts > start_ts
    return dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=x0,
        x1=x1 if show else x0,
        y0=0,
        y1=1,
        fillcolor=TAG_COLORS["anomaly"] if show else "rgba(0,0,0,0)",
        line=dict(width=0),
        layer="below",
        editable=False,
        name=PENDING_RANGE_FILL_NAME,
    )


def build_figure(
    df: pd.DataFrame,
    doc: dict[str, Any],
    *,
    metrics: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    title: str | None = None,
    max_metrics: int = 46,
    filter_data: bool = False,
    hover_values: bool = False,
    width: int | None = None,
    max_points: int = 1500,
    show_labels: bool = True,
    color_metrics: list[str] | None = None,
) -> go.Figure:
    """Trend with label overlays. Zoom via xaxis range (keep full data for click/drag)."""
    view = _filter_df(df, start, end) if filter_data else df
    # `metrics=[]` must stay empty — do not treat it as falsy fallback to all cols.
    cols = metric_columns(df) if metrics is None else list(metrics)
    if max_metrics is not None and max_metrics >= 0:
        cols = cols[:max_metrics]
    color_source = list(color_metrics) if color_metrics else list(cols)

    if not len(view):
        series = {}
    elif filter_data:
        series = plot_series(view, cols, max_points)
    else:
        # Visible (+pad) only — avoids stale coarse context on newly panned edges.
        series = plot_series_window(
            view, cols, start, end, max_points, include_context=False
        )

    # Force WebGL traces to replace on every pan/zoom (Scattergl can keep old buffers).
    data_rev = str(doc.get("plmn") or "labeling")
    if start is not None and end is not None:
        data_rev = (
            f"{data_rev}:{pd.to_datetime(start, utc=True).value}:"
            f"{pd.to_datetime(end, utc=True).value}"
        )

    fig = go.Figure()
    for i, col in enumerate(cols):
        x, y = series.get(col, ([], []))
        metric_name = display_metric(col)
        try:
            color_i = color_source.index(col)
        except ValueError:
            color_i = i
        color = SERIES_COLORWAY[color_i % len(SERIES_COLORWAY)]
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                name=metric_name,
                opacity=0.72,
                line=dict(width=1, color=color),
                hovertemplate=(
                    f"<b>{metric_name}</b><br>"
                    "%{x|%Y년 %m월 %d일 %H:%M}<br>"
                    "값=%{y}<extra></extra>"
                ),
                # Stable uid: embedding data_rev remounts every Scattergl on zoom
                # and freezes the browser. datarevision alone refreshes series data.
                uid=col,
            )
        )

    if len(view) and cols:
        top = view[cols].max(axis=1)
        if filter_data:
            idx = minmax_indices(top.to_numpy(), max_points)
        else:
            detail_start, detail_end = detail_window(view, start, end)
            pos = window_positions(view, detail_start, detail_end)
            if pos is None:
                idx = minmax_indices(top.to_numpy(), max_points)
            else:
                idx = pos[minmax_indices(top.to_numpy()[pos], max_points)]
        anchor = dict(
            x=to_plot_times(view["time"].to_numpy()[idx]),
            y=top.to_numpy()[idx],
            mode="markers",
            marker=dict(size=10, opacity=0),
            name="values",
            showlegend=False,
            uid="values",
        )
        if hover_values:
            # Precomputing per-point text is the slow path; opt-in only.
            rows = view.index.to_numpy()[idx]
            anchor["text"] = [ranked_hover_text(view.loc[i], cols) for i in rows]
            anchor["hovertemplate"] = "%{x|%Y년 %m월 %d일 %H:%M}<br>%{text}<extra></extra>"
        else:
            anchor["hoverinfo"] = "skip"
        fig.add_trace(go.Scattergl(**anchor))

    y_ref = view[cols].max(axis=1) if len(view) and cols else None

    if show_labels:
        for item in doc.get("labels", []):
            s_utc = pd.to_datetime(item["start"], utc=True)
            tag = item.get("tag", "anomaly")
            line = TAG_LINE.get(tag, "crimson")
            note = item.get("note") or ""
            label_id = item.get("id", "")
            add_label_indicator(fig, item)

            if y_ref is not None and len(view):
                nearest = (view["time"] - s_utc).abs().idxmin()
                fig.add_trace(
                    go.Scatter(
                        x=[to_plot_time(view.loc[nearest, "time"])],
                        y=[float(y_ref.loc[nearest])],
                        mode="markers",
                        marker=dict(size=10, color=line, symbol="x"),
                        name=f"{tag}:{label_id}",
                        hovertext=f"{label_id} | {tag} | {note}",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Stable uirevision: changing it every zoom remounts all WebGL traces and
    # freezes the browser. datarevision + uid still refresh series data.
    fig.update_layout(
        title=title or f"{display_plmn(str(doc.get('plmn')))} anomaly labeling",
        height=360,
        hovermode="closest",
        dragmode="zoom",
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="시간 (KST)",
        yaxis_title="value",
        uirevision=str(doc.get("plmn") or "labeling"),
        datarevision=data_rev,
        colorway=list(SERIES_COLORWAY),
        plot_bgcolor="#f7f5f1",
        paper_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="monospace",
            align="left",
            namelength=-1,
        ),
    )
    if width:
        fig.update_layout(width=width, autosize=False)
    else:
        fig.update_layout(autosize=True)
    # Box/drag zoom: horizontal only (time axis). Vertical scale via Y buttons.
    fig.update_yaxes(fixedrange=True)
    # X axis always bounded by this dataset's start/end (not wall-clock "now").
    tmin, tmax = data_time_bounds(view if len(view) else df)
    x0 = pd.to_datetime(start, utc=True) if start is not None else tmin
    x1 = pd.to_datetime(end, utc=True) if end is not None else tmax
    x0, x1 = clamp_time_range(x0, x1, tmin, tmax)
    fig.update_xaxes(
        range=[to_plot_time(x0), to_plot_time(x1)],
        autorange=False,
        type="date",
        tickformat="%Y년 %m월 %d일\n%H:%M",
        hoverformat="%Y년 %m월 %d일 %H:%M",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="royalblue",
        spikethickness=1,
    )
    return fig


def build_metric_figure(
    df: pd.DataFrame,
    doc: dict[str, Any],
    metric: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    filter_data: bool = False,
    width: int | None = None,
    max_points: int = 3000,
) -> go.Figure:
    view = _filter_df(df, start, end) if filter_data else df
    if not len(view):
        idx = []
    elif filter_data:
        idx = minmax_indices(view[metric].to_numpy(), max_points)
    else:
        idx = window_indices(
            view[metric].to_numpy(), window_positions(view, start, end), max_points
        )
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=to_plot_times(view["time"].to_numpy()[idx]),
            y=view[metric].to_numpy()[idx],
            mode="lines",
            name=display_metric(metric),
            line=dict(width=1.5, color="steelblue"),
            hovertemplate="%{x|%Y년 %m월 %d일 %H:%M}<br>%{y}<extra></extra>",
        )
    )
    for item in doc.get("labels", []):
        metrics = item.get("metrics") or ["ALL"]
        if "ALL" not in metrics and metric not in metrics:
            continue
        add_label_indicator(fig, item)
    fig.update_layout(
        title=f"{display_plmn(str(doc.get('plmn')))} — {display_metric(metric)}",
        height=216,
        dragmode="zoom",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="시간 (KST)",
        yaxis_title=display_metric(metric),
        uirevision=f"{doc.get('plmn')}-{metric}",
    )
    if width:
        fig.update_layout(width=width, autosize=False)
    else:
        fig.update_layout(autosize=True)
    fig.update_yaxes(fixedrange=True)
    tmin, tmax = data_time_bounds(view if len(view) else df)
    x0 = pd.to_datetime(start, utc=True) if start is not None else tmin
    x1 = pd.to_datetime(end, utc=True) if end is not None else tmax
    x0, x1 = clamp_time_range(x0, x1, tmin, tmax)
    fig.update_xaxes(
        range=[to_plot_time(x0), to_plot_time(x1)],
        autorange=False,
        type="date",
        tickformat="%Y년 %m월 %d일\n%H:%M",
        hoverformat="%Y년 %m월 %d일 %H:%M",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="royalblue",
        spikethickness=1,
    )
    return fig


def suggest_zscore_points(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Optional reference: timestamps where any metric has |z| > threshold."""
    cols = metric_columns(df)
    mask = pd.Series(False, index=df.index)
    hit_metrics = [[] for _ in range(len(df))]
    for col in cols:
        std = df[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            continue
        z = ((df[col] - df[col].mean()) / std).abs() > threshold
        for i in df.index[z]:
            hit_metrics[i].append(col)
        mask |= z
    out = df.loc[mask, ["time"]].copy()
    out["time"] = [format_kst(t) for t in out["time"]]
    out["n_metrics"] = [len(hit_metrics[i]) for i in out.index]
    out["metrics"] = [
        ", ".join(display_metric(m) for m in hit_metrics[i][:8]) for i in out.index
    ]
    return out.reset_index(drop=True)
