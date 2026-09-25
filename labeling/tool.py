"""Anomaly labeling helpers: ranking, data load, label store, plotting."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ib_data", "raw")
IB_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ib_data")
LABEL_DIR = os.path.join(IB_DATA_DIR, "labels")
PRED_DIR = os.path.join(IB_DATA_DIR, "predictions")
RANK_CACHE_PATH = os.path.join(LABEL_DIR, "plmn_rank.csv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RANK_SIGNATURE_PATH = os.path.join(CACHE_DIR, "plmn_rank.signature")

ANOMALY_FILL = "rgba(201, 162, 39, 0.28)"
ANOMALY_LINE = "#c9a227"
# OmniAnomaly (and other model) predictions — distinct from human labels.
MODEL_FILL = "rgba(124, 58, 237, 0.22)"
MODEL_LINE = "#7c3aed"
ANOMALY_SCORE_KEY = "__anomaly_score__"
ANOMALY_SCORE_COLOR = "#7c3aed"
ANOMALY_SCORE_THRESHOLD_COLOR = "#c026d3"

# Chronological train / valid / test bands (OmniAnomaly IB split).
# Opacity is high enough to read on top of panel_bg fills (#f7f5f1 etc.).
# Chrono split bands — keep distinct from human anomaly (gold) and model (purple).
SPLIT_TRAIN_BAND = "rgba(14, 165, 233, 0.14)"   # sky blue
SPLIT_VALID_BAND = "rgba(16, 185, 129, 0.14)"   # emerald
SPLIT_TEST_BAND = "rgba(244, 114, 182, 0.12)"   # pink
SPLIT_BOUNDARY = "#0e7490"
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2

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


PLMN_MAPPING_PATH = os.path.join(DATA_DIR, "plmn_mapping.txt")
METRIC_MAPPING_PATH = os.path.join(DATA_DIR, "metric_mapping.txt")

_PLMN_ORIGINAL: dict[str, str] | None = None
_METRIC_ORIGINAL: dict[str, str] | None = None
_MAPPING_ENABLED = True


def set_mapping_enabled(enabled: bool) -> None:
    """Enable/disable mapping lookups (PLMN/metric original names).

    When disabled, display helpers keep masked IDs only — safe for shared/git
    notebooks even if mapping files exist locally under data/.
    """
    global _MAPPING_ENABLED
    _MAPPING_ENABLED = bool(enabled)


def mapping_enabled() -> bool:
    return _MAPPING_ENABLED


def _load_mapping_tables() -> None:
    """Lazy-load mapping files if present under data/."""
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
    """Force re-read mapping files (e.g. after replacing data/*_mapping.txt)."""
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


M658_COL = "M658"
M696_COL = "M696"
M971_COL = "M971"
S_RATE_KEY = "S_RATE"
A_RATE_KEY = "A_RATE"
S_RATE_COLOR = "#00BFFF"
A_RATE_COLOR = "#e8590c"
REF_LINE_WIDTH = 2
REF_LINE_DASH = "3px,2px"
REF_LINE = dict(width=REF_LINE_WIDTH, dash=REF_LINE_DASH)
RATE_METRICS = frozenset({S_RATE_KEY, A_RATE_KEY})
RATE_Y_RANGE = (0.0, 1.0)

# Dropped from *all* model training / Argos axes (still shown in labeling UI).
# Almost absent on train, sparse on valid/test, not operationally meaningful.
EXCLUDED_TRAIN_METRICS: frozenset[str] = frozenset({"M688"})


def is_rate_metric(metric: str) -> bool:
    return metric in RATE_METRICS


def filter_train_metrics(metrics: list[str] | tuple[str, ...] | None) -> list[str]:
    """Drop ``EXCLUDED_TRAIN_METRICS`` while preserving order."""
    if not metrics:
        return []
    return [m for m in metrics if m not in EXCLUDED_TRAIN_METRICS]


def rate_metrics_available(
    df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
) -> list[str]:
    """S_RATE / A_RATE keys that exist in the dataset metric list."""
    cols: set[str] = set(metrics or [])
    if df is not None:
        cols.update(metric_columns(df))
    return [key for key in (S_RATE_KEY, A_RATE_KEY) if key in cols]


def overlay_metrics_available(
    df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
) -> list[str]:
    """Synthetic toggles in the metric filter (rate overlays)."""
    return list(rate_metrics_available(df, metrics))


def metric_division_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.astype("float64")
    den = denominator.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def add_rate_columns(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Append S_RATE / A_RATE when inputs exist (skip if already present)."""
    if force or S_RATE_KEY not in df.columns:
        if M658_COL in df.columns and M971_COL in df.columns:
            df[S_RATE_KEY] = metric_division_rate(df[M658_COL], df[M971_COL])
    if force or A_RATE_KEY not in df.columns:
        if M696_COL in df.columns and M971_COL in df.columns:
            df[A_RATE_KEY] = metric_division_rate(df[M696_COL], df[M971_COL])
    return df


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
                # Rates are stored in the pickle; only backfill older caches.
                return add_rate_columns(cached)
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
    df = add_rate_columns(df, force=True)
    df.attrs["signature"] = signature

    if use_cache:
        try:
            df.to_pickle(cache_path)
        except Exception:
            pass
    return df


def display_training_feature(
    metric: str,
    *,
    feature_mode: str | None = None,
) -> str:
    """Label for model inputs (comb_share: counter÷M971 except M971 itself)."""
    name = display_metric(metric)
    if (
        feature_mode == "comb_share"
        and metric != M971_COL
        and not is_rate_metric(metric)
    ):
        return f"{name} ÷ M971"
    return name


def display_metric(metric: str) -> str:
    """M971 or M971 (original) when a local mapping file exists."""
    if metric == S_RATE_KEY:
        return "SUCCESS rate"
    if metric == A_RATE_KEY:
        return "ACCEPT rate"
    original = metric_original(metric)
    return f"{metric} ({original})" if original else metric


def window_plot_indices(
    values: np.ndarray,
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    max_points: int,
) -> np.ndarray:
    """Dataset ends + high-res visible window — same sampling as plot_series_window."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)
    detail_start, detail_end = detail_window(df, start, end)
    pos = window_positions(df, detail_start, detail_end)
    ends = np.array([0, n - 1], dtype=int)
    if pos is None:
        return minmax_indices(values, max_points)
    return np.union1d(ends, pos[minmax_indices(values[pos], max_points)])


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


def chrono_split_bounds(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None = None,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> dict[str, pd.Timestamp] | None:
    """Resolve train / valid / test UTC bounds for the IB chronological split.

    Prefer ``predictions['split']`` from ``run_plmn`` export; otherwise use the
    same default ratios as OmniAnomaly (0.6 / 0.2 / 0.2).
    """
    if df is None or not len(df):
        return None
    times = pd.to_datetime(df["time"], utc=True)
    t_min = times.iloc[0]
    t_max = times.iloc[-1]

    split = (predictions or {}).get("split") or {}
    v0 = split.get("valid_time_start")
    t0 = split.get("test_time_start")
    if v0 and t0:
        valid_start = pd.to_datetime(v0, utc=True)
        test_start = pd.to_datetime(t0, utc=True)
        train_end = split.get("train_time_end")
        valid_end = split.get("valid_time_end")
        return {
            "t_min": t_min,
            "train_end": pd.to_datetime(train_end, utc=True) if train_end else valid_start,
            "valid_start": valid_start,
            "valid_end": pd.to_datetime(valid_end, utc=True) if valid_end else test_start,
            "test_start": test_start,
            "t_max": t_max,
        }

    n = len(df)
    if n < 3:
        return None
    train_end_i = max(int(n * train_ratio), 1)
    valid_end_i = max(int(n * (train_ratio + valid_ratio)), train_end_i + 1)
    valid_end_i = min(valid_end_i, n - 1)
    if valid_end_i <= train_end_i:
        return None
    return {
        "t_min": t_min,
        "train_end": times.iloc[train_end_i - 1],
        "valid_start": times.iloc[train_end_i],
        "valid_end": times.iloc[valid_end_i - 1],
        "test_start": times.iloc[valid_end_i],
        "t_max": t_max,
    }


def add_split_region_overlays(
    fig: go.Figure,
    bounds: dict[str, pd.Timestamp] | None,
) -> None:
    """Draw TRAIN / VALID / TEST bands + dashed boundaries (behind series)."""
    if not bounds:
        return
    t_min = bounds["t_min"]
    valid_start = bounds["valid_start"]
    test_start = bounds["test_start"]
    t_max = bounds["t_max"]

    def _band(x0, x1, fill: str) -> dict[str, Any]:
        return dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=to_plot_time(x0),
            x1=to_plot_time(x1),
            y0=0,
            y1=1,
            fillcolor=fill,
            line=dict(width=0),
            layer="below",
            editable=False,
        )

    def _vline(x) -> dict[str, Any]:
        xp = to_plot_time(x)
        return dict(
            type="line",
            xref="x",
            yref="paper",
            x0=xp,
            x1=xp,
            y0=0,
            y1=1,
            line=dict(color=SPLIT_BOUNDARY, width=3.0, dash="dash"),
            layer="above",
            editable=False,
        )

    def _label(x0, x1, text: str, bgcolor: str, border: str) -> dict[str, Any]:
        mid = x0 + (x1 - x0) / 2
        return dict(
            x=to_plot_time(mid),
            y=1.0,
            xref="x",
            yref="paper",
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=11, color="#1e293b"),
            bgcolor=bgcolor,
            bordercolor=border,
            borderwidth=1,
            borderpad=3,
            yanchor="bottom",
        )

    new_shapes = [
        _band(t_min, valid_start, SPLIT_TRAIN_BAND),
        _band(valid_start, test_start, SPLIT_VALID_BAND),
        _band(test_start, t_max, SPLIT_TEST_BAND),
        _vline(valid_start),
        _vline(test_start),
    ]
    new_ann = [
        _label(t_min, valid_start, "TRAIN", "rgba(241, 245, 249, 0.9)", "#94a3b8"),
        _label(valid_start, test_start, "VALID", "rgba(237, 233, 254, 0.92)", "#8b5cf6"),
        _label(test_start, t_max, "TEST", "rgba(254, 249, 195, 0.92)", "#ca8a04"),
    ]
    existing_shapes = list(fig.layout.shapes or ())
    existing_ann = list(fig.layout.annotations or ())
    # Append after panel_bg (same layer=below) so tints are visible on top of fills.
    fig.layout.shapes = tuple(existing_shapes + new_shapes)
    fig.layout.annotations = tuple(new_ann + existing_ann)


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
    else:
        raise FileNotFoundError(f"No source CSV or ranking cache found under {DATA_DIR}")

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
    return [
        c
        for c in df.columns
        if c not in ("time", "PLMN", ANOMALY_SCORE_KEY)
    ]


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
    for item in doc["labels"]:
        if isinstance(item, dict):
            item.pop("metrics", None)
            item.pop("tag", None)
            item.update(
                normalize_label_reason_fields(
                    reasons=item.get("reasons"),
                    fail_metrics=item.get("fail_metrics"),
                    note=item.get("note"),
                )
            )
    return doc


def save_labels(doc: dict[str, Any]) -> str:
    doc = dict(doc)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    cleaned = []
    for item in doc.get("labels", []):
        item = dict(item)
        item.pop("metrics", None)
        item.pop("tag", None)
        item.update(
            normalize_label_reason_fields(
                reasons=item.get("reasons"),
                fail_metrics=item.get("fail_metrics"),
                note=item.get("note"),
            )
        )
        cleaned.append(item)
    doc["labels"] = cleaned
    path = label_path(doc["plmn"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def pred_path(plmn: str, source: str = "omnianomaly") -> str:
    return os.path.join(PRED_DIR, f"{plmn}_{source}.json")


def list_prediction_sources(plmn: str) -> list[dict[str, str]]:
    """Dropdown options for available model prediction JSON files."""
    options: list[dict[str, str]] = []
    if not os.path.isdir(PRED_DIR):
        return [
            {
                "label": "OmniAnomaly · paperib · 전체 raw · win=100 (파일 없음)",
                "value": "omnianomaly",
            }
        ]
    prefix = f"{plmn}_"
    for name in sorted(os.listdir(PRED_DIR)):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        source = name[len(prefix) : -len(".json")]
        path = os.path.join(PRED_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            doc = {}
        # Listing only needs metadata — drop heavy score_series ASAP.
        doc.pop("score_series", None)
        is_at = source.startswith("anomalytransformer") or doc.get(
            "model"
        ) == "anomalytransformer"
        is_argos = (
            source.startswith("argos")
            or str(doc.get("source") or "").startswith("argos")
            or str(doc.get("model") or "") == "argos"
        )
        is_classical = (
            source.startswith("classical")
            or doc.get("model") == "classical"
        )
        is_lstm = (
            source.startswith("lstm_ad")
            or doc.get("model") == "lstm_ad"
        )
        n_seg = len(doc.get("labels") or [])

        if is_argos:
            # Independent of OmniAnomaly / AT — rule-based overlay.
            mode_tag = "heuristic" if "heuristic" in source else (
                str((doc.get("metrics") or {}).get("mode") or doc.get("mode") or "rule")
            )
            is_ens = (
                "ensemble" in source
                or doc.get("feature_mode") == "argos_ensemble"
                or (doc.get("metrics") or {}).get("ensemble")
            )
            if is_ens:
                axes = doc.get("ensemble_metrics") or (
                    (doc.get("metrics") or {}).get("ensemble_metrics") or []
                )
                fuse = (doc.get("metrics") or {}).get("fuse") or "or"
                n_axes = len(axes) if axes else "?"
                label = (
                    f"Argos · {mode_tag} · ensemble[{fuse} · {n_axes} axes] "
                    f"({n_seg}구간)"
                )
            else:
                metric = doc.get("metric") or "metric?"
                label = f"Argos · {mode_tag} · {metric} ({n_seg}구간)"
            family = 2
            view_ord = 0 if is_ens else 1
            win = 0
        elif is_classical or is_lstm:
            run = doc.get("run_name") or "paperib"
            if is_classical:
                label = f"Classical · IQR+EWMA · {run} ({n_seg}구간)"
                family = 3
            else:
                try:
                    win = int(doc["window"]) if doc.get("window") is not None else int(
                        (doc.get("metrics") or {}).get("window") or 32
                    )
                except (TypeError, ValueError):
                    win = 32
                label = f"LSTM-AD · forecast residual · {run} · win={win} ({n_seg}구간)"
                family = 4
            view_ord = {"paperib": 0, "comb": 1, "comb_share": 2}.get(str(run), 9)
            if is_classical:
                win = 0
        else:
            # OmniAnomaly (default) or Anomaly Transformer only.
            model_name = "Anomaly Transformer" if is_at else "OmniAnomaly"
            if not is_at and not (
                source.startswith("omnianomaly")
                or source == "omnianomaly"
                or doc.get("model") in (None, "", "omnianomaly")
            ):
                # Unknown model file — show its model/source id, never fake OA.
                model_name = str(doc.get("model") or source)
            run = doc.get("run_name")
            if not run:
                stem = source
                if stem.startswith("anomalytransformer_"):
                    stem = stem[len("anomalytransformer_") :]
                elif stem == "anomalytransformer":
                    stem = "paperib"
                elif stem.startswith("omnianomaly_"):
                    stem = stem[len("omnianomaly_") :]
                elif stem == "omnianomaly":
                    stem = "paperib"
                # Strip non-default window suffix: comb_w200 → comb
                run = re.sub(r"_w\d+$", "", stem) or "paperib"
            cols = doc.get("feature_columns") or []
            mode = doc.get("feature_mode") or (
                "all"
                if run == "paperib"
                else "selected"
            )
            # Existing exports used window_length=100; default so older JSON still shows it.
            try:
                win = int(doc["window_length"]) if doc.get("window_length") is not None else 100
            except (TypeError, ValueError):
                win = 100
            if mode == "all" or run == "paperib":
                feat = "전체 raw metrics"
            elif mode == "comb_share" or run == "comb_share":
                feat = "M971 + other÷M971"
            elif mode == "comb" or run == "comb":
                feat = "selected counters + S_RATE/A_RATE"
            else:
                preview = ",".join(str(c) for c in cols[:4])
                if len(cols) > 4:
                    preview = f"{preview},…"
                feat = f"선별 ({preview})"
            label = (
                f"{model_name} · {run} · {feat} · win={win} "
                f"({len(cols)}d, {n_seg}구간)"
            )
            family = 0 if not is_at else 1
            view_ord = {"paperib": 0, "comb": 1, "comb_share": 2}.get(str(run), 9)
        options.append(
            {
                "label": label,
                "value": source,
                "_sort": (family, view_ord, win, source),
            }
        )
    options.sort(key=lambda o: o.pop("_sort"))
    if not options:
        options.append(
            {
                "label": "OmniAnomaly · paperib · 전체 raw · win=100 (파일 없음)",
                "value": "omnianomaly",
            }
        )
    return options


def empty_pred_doc(plmn: str, source: str = "omnianomaly") -> dict[str, Any]:
    return {
        "plmn": plmn,
        "source": source,
        "updated_at": None,
        "threshold": None,
        "labels": [],
    }


def load_predictions(
    plmn: str,
    *,
    source: str = "omnianomaly",
) -> dict[str, Any]:
    """Load model prediction overlay (separate from human label JSON)."""
    path = pred_path(plmn, source=source)
    if not os.path.isfile(path):
        return empty_pred_doc(plmn, source=source)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc.setdefault("plmn", plmn)
    doc.setdefault("source", source)
    doc.setdefault("labels", [])
    _migrate_anomalytransformer_score_scale(doc)
    # Build score index once at load (never re-parse ISO strings per hover/row).
    _ensure_score_index(doc)
    return doc


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN → None


def human_anomaly_mask(
    df: pd.DataFrame,
    labels: list[dict[str, Any]] | None,
) -> np.ndarray:
    """True where a human range/point anomaly covers the sample time."""
    mask = np.zeros(len(df), dtype=bool)
    if not labels or not len(df):
        return mask
    times = pd.to_datetime(df["time"], utc=True)
    for item in labels:
        start = pd.to_datetime(item["start"], utc=True)
        end = pd.to_datetime(item.get("end") or item["start"], utc=True)
        mask |= (times >= start) & (times <= end)
    return mask


def _eval_region_mask(df: pd.DataFrame, predictions: dict[str, Any]) -> np.ndarray:
    """Mask for the scored eval split (valid window from export, else finite scores)."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool)
    times = pd.to_datetime(df["time"], utc=True)
    split = predictions.get("split") or {}
    v0 = split.get("valid_time_start")
    v1 = split.get("valid_time_end")
    if v0 and v1:
        t0 = pd.to_datetime(v0, utc=True)
        t1 = pd.to_datetime(v1, utc=True)
        region = (times >= t0) & (times <= t1)
    else:
        region = np.ones(n, dtype=bool)
    if ANOMALY_SCORE_KEY in df.columns:
        scored = np.isfinite(df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64))
        region = region & scored
    return np.asarray(region, dtype=bool)


def _point_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    tp = int(np.sum(pred & actual))
    tn = int(np.sum(~pred & ~actual))
    fp = int(np.sum(pred & ~actual))
    fn = int(np.sum(~pred & actual))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def adjust_predicts(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """OmniAnomaly point-adjustment (segment hit → fill whole GT segment)."""
    predict = np.asarray(pred, dtype=bool).copy()
    actual = np.asarray(label, dtype=bool)
    anomaly_state = False
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if not actual[j]:
                    break
                predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict


def compute_detection_metrics(
    score: np.ndarray,
    y_true: np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    """Point-wise and point-adjusted P/R/F1 (score < thr ⇒ anomaly)."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    n = min(len(score), len(y))
    score, y = score[:n], y[:n]
    out: dict[str, Any] = {
        "threshold": threshold,
        "n_scored": int(np.isfinite(score).sum()),
        "n_gt": int(y.sum()),
    }
    if threshold is None or not np.isfinite(score).any():
        out["point"] = None
        out["point_adjust"] = None
        return out
    ok = np.isfinite(score)
    pred = np.zeros(n, dtype=bool)
    pred[ok] = score[ok] < float(threshold)
    if ok.any():
        first = int(np.flatnonzero(ok)[0])
        pred_e, y_e = pred[first:], y[first:]
    else:
        pred_e, y_e = pred, y
    out["point"] = _point_metrics(pred_e, y_e)
    out["point_adjust"] = _point_metrics(adjust_predicts(pred_e, y_e), y_e)
    out["n_pred"] = int(pred_e.sum())
    return out


def best_f1_threshold(
    score: np.ndarray,
    y_true: np.ndarray,
    *,
    n_steps: int = 200,
) -> tuple[float | None, dict[str, Any] | None]:
    """Oracle threshold maximizing point-adjusted F1 on the given window."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    n = min(len(score), len(y))
    score, y = score[:n], y[:n]
    ok = np.isfinite(score)
    if not ok.any() or not y[ok].any():
        return None, None
    s = score[ok]
    yy = y[ok]
    lo, hi = float(np.min(s)), float(np.max(s))
    if not np.isfinite(lo) or lo >= hi:
        thr = lo if np.isfinite(lo) else None
        live = compute_detection_metrics(s, yy, thr) if thr is not None else None
        return thr, live
    best_f1 = -1.0
    best_thr = lo
    best_live: dict[str, Any] | None = None
    for thr in np.linspace(lo, hi, max(2, int(n_steps))):
        live = compute_detection_metrics(s, yy, float(thr))
        f1 = float((live.get("point_adjust") or {}).get("f1") or -1.0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_live = live
    return best_thr, best_live


def is_argos_prediction(predictions: dict[str, Any] | None) -> bool:
    """True for Argos / IB heuristic rule overlays (not OA/AT score models)."""
    if not predictions:
        return False
    src = str(predictions.get("source") or "")
    model = str(predictions.get("model") or "")
    method = str(predictions.get("threshold_method") or "")
    return (
        src.startswith("argos")
        or model == "argos"
        or method == "argos_rule"
    )


def is_anomalytransformer_prediction(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    """True for Anomaly Transformer overlays (percentile threshold, not POT)."""
    src = str(source or (predictions or {}).get("source") or "")
    model = str((predictions or {}).get("model") or "")
    method = str((predictions or {}).get("threshold_method") or "")
    return (
        src.startswith("anomalytransformer")
        or model == "anomalytransformer"
    )


def is_baseline_percentile_prediction(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    """Classical / LSTM-AD (train percentile thr, not POT)."""
    src = str(source or (predictions or {}).get("source") or "")
    model = str((predictions or {}).get("model") or "")
    return (
        src.startswith("classical")
        or src.startswith("lstm_ad")
        or model in ("classical", "lstm_ad")
    )


def primary_threshold_label(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> dict[str, str]:
    """UI copy for the primary (``pot``-slot) threshold mode."""
    src = source or (predictions or {}).get("source")
    if is_baseline_percentile_prediction(predictions, source=src):
        model = str((predictions or {}).get("model") or "")
        if str(src).startswith("classical") or model == "classical":
            return {
                "short": "percentile",
                "title": "[percentile]",
                "radio": "percentile (train · classical)",
                "note": (
                    "primary · train score percentile on −#fires; "
                    "binary overlay = OR(IQR,EWMA) (not POT)"
                ),
                "heading": "Detection metrics — classical IQR+EWMA (valid)",
                "overlay": "percentile / OR",
            }
        return {
            "short": "percentile",
            "title": "[percentile]",
            "radio": "percentile (train residual)",
            "note": (
                "primary · LSTM-AD train residual percentile on −MSE (not POT)"
            ),
            "heading": "Detection metrics — LSTM-AD percentile vs best-F1 (valid, PA)",
            "overlay": "percentile",
        }
    if is_anomalytransformer_prediction(predictions, source=src):
        ratio = (predictions or {}).get("anormly_ratio")
        if ratio is None:
            ratio = ((predictions or {}).get("metrics") or {}).get("anormly_ratio")
        try:
            ratio_s = f"{float(ratio):g}%"
        except (TypeError, ValueError):
            ratio_s = "1%"
        return {
            "short": "percentile",
            "title": "[percentile]",
            "radio": f"percentile (train energy · {ratio_s})",
            "note": (
                f"primary · train energy percentile "
                f"(100 − anormly_ratio={ratio_s}; not POT)"
            ),
            "heading": "Detection metrics — percentile vs best-F1 (valid, PA)",
            "overlay": "percentile",
        }
    return {
        "short": "POT",
        "title": "[POT]",
        "radio": "POT (논문 primary)",
        "note": "primary · POT on train scores (SPOT; 오버레이 기본)",
        "heading": "Detection metrics — POT vs best-F1 (valid, PA)",
        "overlay": "POT",
    }


def paper_metrics_from_argos(
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map Argos ``metrics`` (point / event) for the labeling UI."""
    m = metrics or {}
    point = m.get("point") if isinstance(m.get("point"), dict) else None
    event = m.get("event") if isinstance(m.get("event"), dict) else None
    if point is None and _as_float(m.get("f1")) is not None:
        point = m
    out: dict[str, Any] = {
        "mode": m.get("mode"),
        "metric": m.get("metric"),
        "eval_split": m.get("eval_split") or "valid",
        "ensemble": bool(m.get("ensemble")),
        "fuse": m.get("fuse"),
        "ensemble_metrics": m.get("ensemble_metrics"),
    }
    if point:
        out["point"] = {
            "precision": _as_float(point.get("precision")),
            "recall": _as_float(point.get("recall")),
            "f1": _as_float(point.get("f1")),
            "TP": point.get("tp", point.get("TP")),
            "FP": point.get("fp", point.get("FP")),
            "FN": point.get("fn", point.get("FN")),
            "TN": point.get("tn", point.get("TN")),
        }
    if event:
        out["event"] = {
            "event_recall": _as_float(event.get("event_recall")),
            "n_gt_events": event.get("n_gt_events"),
            "n_hit_events": event.get("n_hit_events"),
            "n_pred_events": event.get("n_pred_events"),
        }
    return out


def paper_metrics_from_ib_saved(metrics: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map Omni/AT ``metrics`` block → pot / best_f1 UI blocks."""
    m = metrics or {}
    out: dict[str, dict[str, Any]] = {}
    # OmniAnomaly IB export: valid-pot-*
    if _as_float(m.get("valid-pot-f1")) is not None or _as_float(m.get("pot-threshold")) is not None:
        out["pot"] = {
            "precision": _as_float(m.get("valid-pot-precision")),
            "recall": _as_float(m.get("valid-pot-recall")),
            "f1": _as_float(m.get("valid-pot-f1")),
            "TP": m.get("valid-pot-TP"),
            "FP": m.get("valid-pot-FP"),
            "FN": m.get("valid-pot-FN"),
            "threshold": _as_float(m.get("valid-pot-threshold") or m.get("pot-threshold")),
            "note": "primary · POT from train scores (valid eval)",
        }
    # OmniAnomaly optional best-F1 (bf_search)
    if _as_float(m.get("valid-bf-f1")) is not None or _as_float(m.get("valid-bf-threshold")) is not None:
        out["best_f1"] = {
            "precision": _as_float(m.get("valid-bf-precision")),
            "recall": _as_float(m.get("valid-bf-recall")),
            "f1": _as_float(m.get("valid-bf-f1")),
            "threshold": _as_float(m.get("valid-bf-threshold")),
            "note": "oracle · best-F1 on valid labels",
        }
    # AnomalyTransformer: single valid-f1 → primary (``pot``) slot; UI labels it percentile.
    if "pot" not in out and _as_float(m.get("valid-f1")) is not None:
        ratio = m.get("anormly_ratio")
        try:
            ratio_note = f"anormly_ratio={float(ratio):g}%"
        except (TypeError, ValueError):
            ratio_note = "train energy percentile"
        out["pot"] = {
            "precision": _as_float(m.get("valid-precision")),
            "recall": _as_float(m.get("valid-recall")),
            "f1": _as_float(m.get("valid-f1")),
            "threshold": _as_float(m.get("energy_threshold") or m.get("threshold")),
            "note": f"AnomalyTransformer · {ratio_note} (not POT)",
        }
    return out


def thresholds_from_pred_doc(predictions: dict[str, Any]) -> dict[str, float | None]:
    """Resolve primary / best_f1 thresholds from a prediction JSON.

    Primary is stored under the ``pot`` key for both OA (POT) and AT (percentile).
    """
    m = predictions.get("metrics") or {}
    ss = predictions.get("score_series") or {}
    method = str(predictions.get("threshold_method") or "pot")
    pot = (
        _as_float(m.get("pot-threshold"))
        or _as_float(m.get("valid-pot-threshold"))
        or (
            _as_float(predictions.get("threshold"))
            if method in ("pot", "percentile", "energy_percentile")
            else None
        )
        or _as_float(ss.get("threshold"))
        or _as_float(predictions.get("threshold"))
    )
    best = _as_float(m.get("valid-bf-threshold"))
    # AT energy-scale: display thr is often 0.0 — still treat as primary.
    if pot is None and predictions.get("score_scale") == "log10_thr_over_energy":
        pot = _as_float(predictions.get("threshold"))
        if pot is None:
            pot = 0.0
    if pot is None and method in ("percentile", "energy_percentile"):
        pot = _as_float(m.get("energy_threshold"))
        if pot is None:
            pot = _as_float(predictions.get("threshold"))
            if pot is None:
                pot = 0.0
    return {"pot": pot, "best_f1": best}


def mask_to_pred_labels(
    mask: np.ndarray,
    times: pd.Series | np.ndarray,
    *,
    scores: np.ndarray | None = None,
    id_prefix: str = "pred",
    source: str = "omnianomaly",
) -> list[dict[str, Any]]:
    """Boolean mask → UI prediction segments (no attribution)."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    times = pd.to_datetime(pd.Series(times), utc=True)
    n = min(len(mask), len(times))
    items: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        seg_score = None
        if scores is not None:
            sl = np.asarray(scores[i:j], dtype=np.float64)
            finite = sl[np.isfinite(sl)]
            if len(finite):
                seg_score = float(np.min(finite))
        kind = "point" if j == i + 1 else "range"
        items.append(
            {
                "id": f"{id_prefix}_{uuid.uuid4().hex[:8]}",
                "kind": kind,
                "start": times.iloc[i].isoformat(),
                "end": times.iloc[j - 1].isoformat(),
                "score": seg_score,
                "source": source,
            }
        )
        i = j
    return items


def rebuild_pred_labels_for_threshold(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    threshold: float,
    *,
    id_prefix: str = "pred",
) -> list[dict[str, Any]]:
    """Rebuild prediction segments from aligned anomaly scores at ``threshold``."""
    if ANOMALY_SCORE_KEY not in df.columns or not len(df):
        return []
    score = df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64)
    ok = np.isfinite(score)
    mask = np.zeros(len(df), dtype=bool)
    mask[ok] = score[ok] < float(threshold)
    return mask_to_pred_labels(
        mask,
        df["time"],
        scores=score,
        id_prefix=id_prefix,
        source=str(predictions.get("source") or "omnianomaly"),
    )


def enrich_predictions_for_ui(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    human_labels: list[dict[str, Any]] | None,
    *,
    threshold_mode: str = "pot",
) -> dict[str, Any]:
    """Attach dual-threshold metrics + mode-specific labels for the labeling UI.

    Keeps original file labels (with attribution) for POT when the export used POT.
    Rebuilds best-F1 segments from the score series; searches thr if missing.
    Argos rule overlays skip score thresholds and expose point/event metrics.
    """
    doc = dict(predictions)
    if is_argos_prediction(doc):
        file_labels = list(doc.get("labels") or [])
        paper = paper_metrics_from_argos(doc.get("metrics") or {})
        doc["threshold"] = None
        doc["threshold_mode"] = "rule"
        doc["thresholds"] = {"pot": None, "best_f1": None}
        doc["paper_metrics"] = paper
        doc["live_by_mode"] = {}
        doc["labels_by_mode"] = {"rule": file_labels, "pot": file_labels, "best_f1": []}
        doc["labels"] = file_labels
        doc["n_pred_segments"] = len(file_labels)
        doc["n_pred_points"] = int(
            sum(
                1
                if it.get("kind") == "point"
                else max(
                    1,
                    int(
                        (
                            pd.to_datetime(it.get("end"), utc=True)
                            - pd.to_datetime(it.get("start"), utc=True)
                        ).total_seconds()
                        / 60
                    )
                    + 1,
                )
                for it in file_labels
            )
        ) if file_labels else 0
        return doc

    mode = threshold_mode if threshold_mode in ("pot", "best_f1") else "pot"
    thresholds = thresholds_from_pred_doc(doc)
    paper = paper_metrics_from_ib_saved(doc.get("metrics") or {})

    region = _eval_region_mask(df, doc)
    y_full = human_anomaly_mask(df, human_labels)
    if ANOMALY_SCORE_KEY in df.columns and region.any():
        score = df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64)
        score_e = np.where(region, score, np.nan)
        y_e = y_full & region
    else:
        score_e = np.array([], dtype=np.float64)
        y_e = np.array([], dtype=bool)

    # Resolve best-F1 thr: saved → live search.
    if thresholds.get("best_f1") is None and len(score_e) and y_e.any():
        bf_thr, _ = best_f1_threshold(score_e[region], y_full[region])
        thresholds["best_f1"] = bf_thr

    live_by_mode: dict[str, Any] = {}
    for mkey, mthr in thresholds.items():
        if mthr is None or not len(score_e):
            continue
        live_by_mode[mkey] = compute_detection_metrics(
            score_e[region], y_full[region], mthr
        )

    # Labels per mode
    labels_by_mode: dict[str, list] = {}
    pot_thr = thresholds.get("pot")
    method = str(doc.get("threshold_method") or "pot")
    file_labels = list(doc.get("labels") or [])
    if pot_thr is not None and method in (
        "pot",
        "percentile",
        "energy_percentile",
    ) and file_labels:
        labels_by_mode["pot"] = file_labels
    elif pot_thr is not None:
        labels_by_mode["pot"] = rebuild_pred_labels_for_threshold(
            df, doc, float(pot_thr), id_prefix="pot"
        )
    else:
        labels_by_mode["pot"] = file_labels

    bf_thr = thresholds.get("best_f1")
    if bf_thr is not None:
        labels_by_mode["best_f1"] = rebuild_pred_labels_for_threshold(
            df, doc, float(bf_thr), id_prefix="bf"
        )
    else:
        labels_by_mode["best_f1"] = []

    if mode not in labels_by_mode or (
        mode == "best_f1" and not labels_by_mode.get("best_f1") and labels_by_mode.get("pot")
    ):
        if thresholds.get(mode) is None:
            mode = "pot" if thresholds.get("pot") is not None else mode

    thr = thresholds.get(mode)
    if thr is None:
        thr = thresholds.get("pot") or thresholds.get("best_f1")
        if thr is not None and mode == "best_f1" and thresholds.get("best_f1") is None:
            mode = "pot"

    active_labels = labels_by_mode.get(mode) or []
    doc["threshold"] = thr
    doc["threshold_mode"] = mode
    doc["thresholds"] = thresholds
    doc["paper_metrics"] = paper
    doc["live_by_mode"] = live_by_mode
    doc["labels_by_mode"] = labels_by_mode
    doc["labels"] = active_labels
    ss = dict(doc.get("score_series") or {})
    if thr is not None:
        ss["threshold"] = float(thr)
    doc["score_series"] = ss
    doc["n_pred_segments"] = len(active_labels)
    doc["n_pred_points"] = int(
        sum(
            1
            if it.get("kind") == "point"
            else max(
                1,
                int(
                    (
                        pd.to_datetime(it.get("end"), utc=True)
                        - pd.to_datetime(it.get("start"), utc=True)
                    ).total_seconds()
                    / 60
                )
                + 1,
            )
            for it in active_labels
        )
    ) if active_labels else 0
    return doc


def apply_threshold_mode(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    *,
    threshold_mode: str,
) -> dict[str, Any]:
    """Switch active overlay labels / threshold without re-searching best-F1."""
    doc = dict(predictions)
    if is_argos_prediction(doc):
        labels = list(doc.get("labels") or (doc.get("labels_by_mode") or {}).get("rule") or [])
        doc["threshold_mode"] = "rule"
        doc["threshold"] = None
        doc["labels"] = labels
        doc["n_pred_segments"] = len(labels)
        return doc
    mode = threshold_mode if threshold_mode in ("pot", "best_f1") else "pot"
    thresholds = dict(doc.get("thresholds") or thresholds_from_pred_doc(doc))
    labels_by_mode = dict(doc.get("labels_by_mode") or {})
    if not labels_by_mode:
        # Not enriched yet — caller should use enrich_predictions_for_ui.
        return enrich_predictions_for_ui(
            df,
            predictions,
            None,
            threshold_mode=mode,
        )
    if mode == "best_f1" and not labels_by_mode.get("best_f1"):
        thr = thresholds.get("best_f1")
        if thr is not None and ANOMALY_SCORE_KEY in df.columns:
            labels_by_mode["best_f1"] = rebuild_pred_labels_for_threshold(
                df, doc, float(thr), id_prefix="bf"
            )
        else:
            mode = "pot"
    thr = thresholds.get(mode)
    if thr is None:
        mode = "pot"
        thr = thresholds.get("pot")
    active = labels_by_mode.get(mode) or []
    doc["threshold_mode"] = mode
    doc["threshold"] = thr
    doc["thresholds"] = thresholds
    doc["labels_by_mode"] = labels_by_mode
    doc["labels"] = active
    ss = dict(doc.get("score_series") or {})
    if thr is not None:
        ss["threshold"] = float(thr)
    doc["score_series"] = ss
    doc["n_pred_segments"] = len(active)
    return doc


def _migrate_anomalytransformer_score_scale(doc: dict[str, Any]) -> None:
    """Rescale legacy AT scores (``-energy`` ≈ 1e-4) to log10(thr/energy)."""
    source = str(doc.get("source") or "")
    is_at = doc.get("model") == "anomalytransformer" or source.startswith(
        "anomalytransformer"
    )
    if not is_at:
        return
    if doc.get("score_scale") == "log10_thr_over_energy":
        return
    ss = doc.get("score_series") or {}
    scores_raw = ss.get("scores") or []
    if not scores_raw:
        return
    energy_thr = doc.get("energy_threshold")
    thr = doc.get("threshold")
    if energy_thr is None and thr is not None:
        # Legacy: UI threshold was -energy_threshold.
        energy_thr = abs(float(thr))
    if energy_thr is None or not (float(energy_thr) > 0):
        return
    energy_thr = float(energy_thr)
    # Legacy polarity: stored score = -energy.
    energies: list[float] = []
    for v in scores_raw:
        if v is None:
            energies.append(float("nan"))
        else:
            energies.append(max(-float(v), 0.0))
    eps = 1e-15
    thr_c = max(energy_thr, eps)
    new_scores: list[float | None] = []
    for e in energies:
        if e != e:  # NaN
            new_scores.append(None)
        else:
            new_scores.append(float(np.log10(thr_c) - np.log10(max(e, eps))))
    ss = dict(ss)
    ss["scores"] = new_scores
    ss["threshold"] = 0.0
    ss["note"] = (
        "score = log10(energy_threshold) - log10(energy); "
        "higher = more normal; anomaly when score < 0"
    )
    doc["score_series"] = ss
    doc["threshold"] = 0.0
    doc["energy_threshold"] = energy_thr
    doc["score_polarity"] = "log10_thr_over_energy"
    doc["score_scale"] = "log10_thr_over_energy"
    # Refresh segment scores to the new scale (min = most anomalous).
    times = ss.get("times") or []
    if times and new_scores:
        t_ns = (
            pd.to_datetime(times, utc=True, errors="coerce")
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64, copy=False)
        )
        sc = np.asarray(
            [np.nan if v is None else float(v) for v in new_scores],
            dtype=np.float64,
        )
        for item in doc.get("labels") or []:
            try:
                s = int(pd.to_datetime(item["start"], utc=True).value)
                e = int(pd.to_datetime(item["end"], utc=True).value)
            except (ValueError, TypeError, KeyError):
                continue
            mask = (t_ns >= s) & (t_ns <= e) & np.isfinite(sc)
            if np.any(mask):
                item["score"] = float(np.min(sc[mask]))
    # Drop stale index so it rebuilds after migration.
    doc.pop("_score_index", None)


def _ensure_score_index(predictions: dict[str, Any] | None) -> dict[str, Any] | None:
    """Cache sorted ns timestamps + scores on the predictions dict."""
    if not predictions:
        return None
    cached = predictions.get("_score_index")
    if isinstance(cached, dict) and "ns" in cached and "scores" in cached:
        return cached
    ss = predictions.get("score_series") or {}
    times_raw = ss.get("times") or []
    scores_raw = ss.get("scores") or []
    if not times_raw or not scores_raw:
        predictions["_score_index"] = {"ns": np.array([], dtype=np.int64), "scores": np.array([], dtype=np.float64)}
        return predictions["_score_index"]
    times = pd.to_datetime(times_raw, utc=True, errors="coerce")
    scores = np.asarray(scores_raw, dtype=np.float64)
    mask = ~pd.isna(times)
    times = times[mask]
    scores = scores[mask]
    if len(times) == 0:
        predictions["_score_index"] = {"ns": np.array([], dtype=np.int64), "scores": np.array([], dtype=np.float64)}
        return predictions["_score_index"]
    # Always nanoseconds since epoch (pandas may store datetime64[us]).
    ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    order = np.argsort(ns, kind="mergesort")
    index = {
        "ns": np.asarray(ns[order], dtype=np.int64),
        "scores": np.asarray(scores[order], dtype=np.float64),
    }
    predictions["_score_index"] = index
    return index


def prediction_score_lookup(
    predictions: dict[str, Any] | None,
) -> dict[pd.Timestamp, float]:
    """Map UTC sample time → anomaly score (compatibility helper).

    Prefer ``align_scores_to_df`` / ``anomaly_score_at_time`` for hot paths.
    """
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0:
        return {}
    out: dict[pd.Timestamp, float] = {}
    for ns, score in zip(index["ns"], index["scores"]):
        ts = pd.Timestamp(int(ns), unit="ns", tz="UTC")
        out[ts] = float(score)
    return out


def align_scores_to_df(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None,
) -> np.ndarray:
    """Float64 score per df row (NaN where unscored). Vectorized, O(n log m)."""
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0 or n == 0 or "time" not in df.columns:
        return out
    sample_ns = (
        pd.to_datetime(df["time"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64, copy=False)
    )
    score_ns = index["ns"]
    scores = index["scores"]
    pos = np.searchsorted(score_ns, sample_ns)
    tol = np.int64(5 * 60 * 1_000_000_000)
    # Candidate indices: pos-1 and pos (clipped).
    left = np.clip(pos - 1, 0, len(score_ns) - 1)
    right = np.clip(pos, 0, len(score_ns) - 1)
    dt_l = np.abs(score_ns[left] - sample_ns)
    dt_r = np.abs(score_ns[right] - sample_ns)
    use_left = dt_l <= dt_r
    best_i = np.where(use_left, left, right)
    best_dt = np.where(use_left, dt_l, dt_r)
    ok = best_dt <= tol
    out[ok] = scores[best_i[ok]]
    return out


def anomaly_score_at_time(
    predictions: dict[str, Any] | None,
    ts,
    *,
    lookup: dict[pd.Timestamp, float] | None = None,
) -> float | None:
    """Score for a sample time (exact / nearest within 5 min)."""
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0:
        return None
    target = pd.to_datetime(ts, utc=True)
    if getattr(target, "tzinfo", None) is None:
        target = target.tz_localize("UTC")
    sn = int(target.value)
    score_ns = index["ns"]
    scores = index["scores"]
    p = int(np.searchsorted(score_ns, sn))
    tol = 5 * 60 * 1_000_000_000
    best = None
    best_dt = None
    for cand in (p - 1, p):
        if 0 <= cand < len(score_ns):
            dt = abs(int(score_ns[cand]) - sn)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = cand
    if best is not None and best_dt is not None and best_dt <= tol:
        return float(scores[best])
    # Optional dict path (legacy callers).
    if lookup:
        if target in lookup:
            return lookup[target]
        floored = target.floor("min")
        if floored in lookup:
            return lookup[floored]
    return None


def score_hover_arrays(
    predictions: dict[str, Any] | None,
) -> tuple[list[int], list[float]] | None:
    """Plot-axis ms + scores for browser hover (same clock as ``sample_ms``)."""
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0:
        return None
    # Rebuild UTC timestamps → plot naive (KST wall) ms, matching sample_ms.
    utc_times = pd.to_datetime(index["ns"], unit="ns", utc=True)
    plot_ms = to_plot_times(utc_times).astype("datetime64[ms]").astype("int64")
    scores = np.asarray(index["scores"], dtype=np.float64)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return None
    return plot_ms[finite].tolist(), scores[finite].tolist()


def score_cache_path(plmn: str, source: str = "omnianomaly") -> str:
    safe = str(source).replace("/", "_")
    return os.path.join(CACHE_DIR, f"{plmn}_score_{safe}.pkl")


def _df_time_ns(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_datetime(df["time"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64, copy=False)
    )


def write_score_cache(
    plmn: str,
    source: str,
    *,
    values: np.ndarray,
    time_ns: np.ndarray,
    threshold: float | None = None,
    pred_mtime: float | None = None,
    score_scale: str | None = None,
) -> str:
    """Persist df-aligned anomaly scores next to the PLMN frame cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = score_cache_path(plmn, source)
    payload = {
        "plmn": plmn,
        "source": source,
        "n_rows": int(len(values)),
        "time_ns": np.asarray(time_ns, dtype=np.int64),
        "values": np.asarray(values, dtype=np.float64),
        "threshold": None if threshold is None else float(threshold),
        "pred_mtime": pred_mtime,
        "score_scale": score_scale,
    }
    pd.to_pickle(payload, path)
    return path


def ensure_score_column(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
    plmn: str | None = None,
) -> pd.DataFrame:
    """Attach ``ANOMALY_SCORE_KEY`` aligned 1:1 with ``df`` (file-cached)."""
    out = df
    if ANOMALY_SCORE_KEY in out.columns:
        out = out.drop(columns=[ANOMALY_SCORE_KEY])
    if not predictions or not len(out):
        return out
    plmn = str(plmn or predictions.get("plmn") or "")
    source = str(source or predictions.get("source") or "omnianomaly")
    score_scale = predictions.get("score_scale")
    if not plmn:
        values = align_scores_to_df(out, predictions)
        out = out.copy()
        out[ANOMALY_SCORE_KEY] = values
        return out

    ppath = pred_path(plmn, source)
    pred_mtime = os.path.getmtime(ppath) if os.path.isfile(ppath) else None
    time_ns = _df_time_ns(out)
    cpath = score_cache_path(plmn, source)
    thr = None
    ss = predictions.get("score_series") or {}
    if ss.get("threshold") is not None:
        thr = float(ss["threshold"])
    elif predictions.get("threshold") is not None:
        thr = float(predictions["threshold"])
    if os.path.isfile(cpath):
        try:
            cached = pd.read_pickle(cpath)
            if (
                cached.get("pred_mtime") == pred_mtime
                and cached.get("score_scale") == score_scale
                and cached.get("threshold") == thr
                and int(cached.get("n_rows") or -1) == len(out)
                and np.array_equal(
                    np.asarray(cached.get("time_ns"), dtype=np.int64), time_ns
                )
            ):
                out = out.copy()
                out[ANOMALY_SCORE_KEY] = np.asarray(cached["values"], dtype=np.float64)
                return out
        except Exception:
            pass

    values = align_scores_to_df(out, predictions)
    try:
        write_score_cache(
            plmn,
            source,
            values=values,
            time_ns=time_ns,
            threshold=thr,
            pred_mtime=pred_mtime,
            score_scale=str(score_scale) if score_scale is not None else None,
        )
    except Exception:
        pass
    out = out.copy()
    out[ANOMALY_SCORE_KEY] = values
    return out


def score_at_row(row: pd.Series) -> float | None:
    """Score from a df row column when present."""
    if ANOMALY_SCORE_KEY not in row.index:
        return None
    try:
        v = float(row[ANOMALY_SCORE_KEY])
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def add_model_indicator(
    fig: go.Figure,
    item: dict[str, Any],
    *,
    highlight: bool = False,
    draw_edges: bool = True,
) -> bool:
    """Draw a model-prediction overlay (purple) — read-only.

    ``draw_edges=False`` skips start/end lines (cheaper when many segments).
    """
    s_utc = pd.to_datetime(item["start"], utc=True)
    e_utc = pd.to_datetime(item["end"], utc=True)
    s = to_plot_time(s_utc)
    e = to_plot_time(e_utc)
    is_range = item.get("kind") == "range" or s_utc != e_utc
    fill = "rgba(124, 58, 237, 0.45)" if highlight else MODEL_FILL
    line = "#4c1d95" if highlight else MODEL_LINE
    edge_w = 3 if highlight else 2
    lid = str(item.get("id") or "")

    def _edge(x, which: str) -> None:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(color=line, width=edge_w, dash="dot"),
            layer="above",
            editable=False,
            name=f"model_edge_{which}:{lid}" if lid else None,
        )

    if is_range:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=s,
            x1=e,
            y0=0,
            y1=1,
            fillcolor=fill,
            line=dict(width=0),
            layer="above",
            editable=False,
            name=f"model_fill:{lid}" if lid else None,
        )
        if draw_edges or highlight:
            _edge(s, "start")
            _edge(e, "end")
    else:
        _edge(s, "start")
    return is_range


# Cap Plotly shapes: Anomaly Transformer can emit hundreds of segments.
MAX_MODEL_OVERLAYS = 48


def model_overlays_for_plot(
    items: list[dict[str, Any]] | None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    highlight_id: str | None = None,
    max_items: int = MAX_MODEL_OVERLAYS,
) -> list[dict[str, Any]]:
    """Pick overlays to draw: in-window, prefer longer/worse, always keep selection."""
    if not items:
        return []
    t0 = pd.to_datetime(start, utc=True) if start is not None else None
    t1 = pd.to_datetime(end, utc=True) if end is not None else None
    hi = str(highlight_id) if highlight_id is not None else None

    ranked: list[tuple[float, float, dict[str, Any]]] = []
    selected_item: dict[str, Any] | None = None
    for item in items:
        try:
            s = pd.to_datetime(item["start"], utc=True)
            e = pd.to_datetime(item["end"], utc=True)
        except (ValueError, TypeError, KeyError):
            continue
        if hi is not None and str(item.get("id") or "") == hi:
            selected_item = item
        if t0 is not None and t1 is not None and (e < t0 or s > t1):
            continue
        dur = max((e - s).total_seconds(), 60.0)
        try:
            scv = float(item["score"]) if item.get("score") is not None else 0.0
        except (TypeError, ValueError):
            scv = 0.0
        # Longer first; then lower UI score (more anomalous).
        ranked.append((dur, -scv, item))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if selected_item is not None:
        sid = str(selected_item.get("id") or "")
        out.append(selected_item)
        if sid:
            seen.add(sid)
    for _, _, item in ranked:
        iid = str(item.get("id") or "")
        if iid and iid in seen:
            continue
        out.append(item)
        if iid:
            seen.add(iid)
        if len(out) >= max_items:
            break
    return out


def labels_to_frame(doc: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in doc.get("labels", []):
        kind = item.get("kind", "point")
        start = format_kst(item.get("start"))
        end = format_kst(item.get("end"))
        interval = start if kind == "point" else f"{start} → {end}"
        rows.append(
            {
                "id": item.get("id"),
                "형식": "점" if kind == "point" else "구간",
                "시각 / 구간 (KST)": interval,
                "판정 기준": label_reason_summary(item),
                "updated_at": format_kst(item.get("updated_at"), seconds=True),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "형식",
                "시각 / 구간 (KST)",
                "판정 기준",
                "updated_at",
            ]
        )
    return pd.DataFrame(rows)


LABEL_REASON_DEFS: list[tuple[str, str]] = [
    ("rate_degrade", "SUCCESS/ACCEPT rate 급락·낮은 지속"),
    ("attempt_drop", "M971 급락"),
    ("attempt_spike", "M971 급증"),
    ("fail_surge", "FAIL/TO 등 실패·타임아웃 폭증"),
    ("other", "기타 (메모 권장)"),
]
LABEL_REASON_CODES = frozenset(code for code, _ in LABEL_REASON_DEFS)
LABEL_REASON_LABELS = dict(LABEL_REASON_DEFS)


def normalize_label_reason_fields(
    *,
    reasons: list[str] | None = None,
    fail_metrics: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Sanitize reason / fail_metrics / note for a human label item."""
    codes = []
    for raw in reasons or []:
        code = str(raw).strip()
        if code in LABEL_REASON_CODES and code not in codes:
            codes.append(code)
    metrics: list[str] = []
    if "fail_surge" in codes:
        for raw in fail_metrics or []:
            m = str(raw).strip()
            if m and m not in metrics:
                metrics.append(m)
    note_s = (note or "").strip()
    out: dict[str, Any] = {"reasons": codes}
    if metrics:
        out["fail_metrics"] = metrics
    else:
        out["fail_metrics"] = []
    out["note"] = note_s
    return out


def label_reason_summary(item: dict[str, Any]) -> str:
    """Short Korean summary of stored reasons for list UI."""
    codes = [c for c in (item.get("reasons") or []) if c in LABEL_REASON_CODES]
    if not codes:
        return "기준 미입력"
    parts: list[str] = []
    for c in codes:
        if c == "fail_surge":
            mets = [str(m) for m in (item.get("fail_metrics") or []) if m]
            if mets:
                shown = ", ".join(mets[:4])
                if len(mets) > 4:
                    shown += "…"
                parts.append(f"실패 폭증({shown})")
            else:
                parts.append(LABEL_REASON_LABELS.get(c, c))
        else:
            parts.append(LABEL_REASON_LABELS.get(c, c))
    note = (item.get("note") or "").strip()
    text = " · ".join(parts)
    if note:
        text = f"{text} · {note[:40]}{'…' if len(note) > 40 else ''}"
    return text


def labels_missing_reasons(doc: dict[str, Any] | None) -> list[str]:
    """Label ids that have no reason codes (should set before save)."""
    missing = []
    for item in (doc or {}).get("labels") or []:
        codes = [c for c in (item.get("reasons") or []) if c in LABEL_REASON_CODES]
        if not codes:
            missing.append(str(item.get("id")))
    return missing


def add_label(
    doc: dict[str, Any],
    *,
    kind: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    label_id: str | None = None,
    reasons: list[str] | None = None,
    fail_metrics: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    kind = kind.lower()
    if kind not in {"point", "range"}:
        raise ValueError("kind must be 'point' or 'range'")

    start_ts = parse_time(start)
    end_ts = start_ts if kind == "point" else parse_time(end)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    item = {
        "id": label_id or str(uuid.uuid4())[:8],
        "kind": kind,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    item.update(
        normalize_label_reason_fields(
            reasons=reasons,
            fail_metrics=fail_metrics,
            note=note,
        )
    )

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
    if "start" in fields:
        found["start"] = parse_time(fields["start"]).isoformat()
    if "end" in fields:
        found["end"] = parse_time(fields["end"]).isoformat()
    if any(k in fields for k in ("reasons", "fail_metrics", "note")):
        found.update(
            normalize_label_reason_fields(
                reasons=fields["reasons"]
                if "reasons" in fields
                else found.get("reasons"),
                fail_metrics=fields["fail_metrics"]
                if "fail_metrics" in fields
                else found.get("fail_metrics"),
                note=fields["note"] if "note" in fields else found.get("note"),
            )
        )
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


def format_metric_value(
    value: float,
    *,
    metric: str | None = None,
    attempt: float | None = None,
) -> str:
    """Format a metric; append (share of M971 %) when possible."""
    if metric and is_rate_metric(metric):
        return f"{float(value) * 100.0:.1f}%"
    base = f"{round(float(value)):,}"
    if attempt is None:
        return base
    try:
        att = float(attempt)
    except (TypeError, ValueError):
        return base
    if not (att == att) or att == 0:  # NaN or zero
        return base
    return f"{base} ({float(value) / att * 100.0:.1f}%)"


# Highlighted in the "이 시점 특성값" panel (rank, name, and value).
HOVER_RED_METRICS = frozenset({"M855", "M037", "M430", "M162", "M618", "M520"})
HOVER_ORANGE_METRICS = frozenset({"M874", "M185", "M965", "M843", "M419"})


# Ranked Plotly tooltips are only embedded while zoomed-in. Full-window
# rebuilds (「전체」) skip them so figure JSON stays small; series keep their
# normal hovertemplates and the bottom panel still ranks on hoverData.
HOVER_RANKED_MAX_FRAC = 0.45


def _row_attempt(row: pd.Series) -> float | None:
    if M971_COL not in row.index:
        return None
    try:
        att = float(row[M971_COL])
    except (TypeError, ValueError):
        return None
    if not (att == att):
        return None
    return att


def _should_embed_ranked_hover(
    df: pd.DataFrame,
    start,
    end,
) -> bool:
    if start is None or end is None or not len(df):
        return False
    tmin = pd.to_datetime(df["time"].min(), utc=True)
    tmax = pd.to_datetime(df["time"].max(), utc=True)
    full = (tmax - tmin).total_seconds()
    if full <= 0:
        return False
    win = (
        pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)
    ).total_seconds()
    return 0 < win < full * HOVER_RANKED_MAX_FRAC


def ranked_hover_text(
    row: pd.Series,
    cols: list[str],
    *,
    sep: str = "<br>",
    top_n: int | None = None,
    extra_pairs: list[tuple[str, float]] | None = None,
) -> str:
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    if extra_pairs:
        pairs = pairs + list(extra_pairs)
        pairs.sort(key=lambda x: x[1], reverse=True)
    attempt = _row_attempt(row)
    shown = pairs if top_n is None else pairs[:top_n]
    lines = [
        f"{display_metric(name)}="
        f"{format_metric_value(val, metric=name, attempt=attempt)}"
        for name, val in shown
    ]
    if top_n is not None and len(pairs) > top_n:
        lines.append(f"... +{len(pairs) - top_n} more (아래 패널 스크롤)")
    return sep.join(lines)


def _ranked_hover_texts_matrix(
    view: pd.DataFrame,
    cols: list[str],
    idx: np.ndarray,
    *,
    top_n: int | None = None,
    extra_cols: list[str] | None = None,
    extra_mat: np.ndarray | None = None,
) -> list[str]:
    """Vectorized ranked hover strings for many samples (avoids per-row Series work)."""
    use_cols = [c for c in cols if c in view.columns]
    n = len(idx)
    if n == 0:
        return []
    if not use_cols and (extra_mat is None or extra_cols is None):
        return ["(값 없음)"] * n

    labels = [display_metric(c) for c in use_cols]
    is_rate = [is_rate_metric(c) for c in use_cols]
    metric_ids = list(use_cols)
    mat = (
        view.loc[:, use_cols].to_numpy(dtype=np.float64, copy=False)[idx]
        if use_cols
        else np.zeros((n, 0), dtype=np.float64)
    )
    attempt_vals = (
        view[M971_COL].to_numpy(dtype=np.float64, copy=False)[idx]
        if M971_COL in view.columns
        else None
    )

    extra_labels: list[str] = []
    extra_is_rate: list[bool] = []
    extra_ids: list[str] = []
    if (
        extra_cols is not None
        and extra_mat is not None
        and len(extra_cols)
        and extra_mat.shape[0] == n
    ):
        extra_labels = [display_metric(c) for c in extra_cols]
        extra_is_rate = [is_rate_metric(c) for c in extra_cols]
        extra_ids = list(extra_cols)

    texts: list[str] = []
    for i in range(n):
        pairs: list[tuple[str, float, bool, str]] = []
        row = mat[i] if mat.shape[1] else None
        if row is not None:
            for j, lab in enumerate(labels):
                v = float(row[j])
                if v == v and v != 0:  # finite & nonzero
                    pairs.append((lab, v, is_rate[j], metric_ids[j]))
        if extra_labels:
            erow = extra_mat[i]
            for j, lab in enumerate(extra_labels):
                v = float(erow[j])
                if v == v and v != 0:
                    pairs.append((lab, v, extra_is_rate[j], extra_ids[j]))
        if not pairs:
            texts.append("(값 없음)")
            continue
        pairs.sort(key=lambda p: p[1], reverse=True)
        total = len(pairs)
        shown = pairs if top_n is None else pairs[:top_n]
        att = float(attempt_vals[i]) if attempt_vals is not None else None
        if att is not None and not (att == att):
            att = None
        lines = []
        for lab, v, rate, mid in shown:
            lines.append(
                f"{lab}={format_metric_value(v, metric=mid, attempt=att)}"
            )
        if top_n is not None and total > top_n:
            lines.append(f"... +{total - top_n} more")
        texts.append("<br>".join(lines))
    return texts


def _hover_sample_indices(
    view: pd.DataFrame,
    y_cols: list[str],
    *,
    start,
    end,
    max_points: int,
    filter_data: bool,
) -> np.ndarray:
    """Downsample indices for an invisible hover hit-target along y_cols max."""
    top = view[y_cols].max(axis=1)
    if filter_data:
        return minmax_indices(top.to_numpy(), max_points)
    detail_start, detail_end = detail_window(view, start, end)
    pos = window_positions(view, detail_start, detail_end)
    if pos is None:
        return minmax_indices(top.to_numpy(), max_points)
    return pos[minmax_indices(top.to_numpy()[pos], max_points)]


def _add_ranked_hover_anchor(
    fig: go.Figure,
    view: pd.DataFrame,
    *,
    cols: list[str],
    row: int,
    uid: str,
    start,
    end,
    max_points: int,
    filter_data: bool,
    extra_cols: list[str] | None = None,
    extra_series: dict[str, pd.Series] | None = None,
    top_n: int | None = None,
) -> None:
    """Invisible markers whose tooltip lists ``cols`` values largest-first."""
    if not len(view) or not cols:
        return
    y_cols = [c for c in cols if c in view.columns]
    if not y_cols:
        return
    idx = _hover_sample_indices(
        view,
        y_cols,
        start=start,
        end=end,
        max_points=max_points,
        filter_data=filter_data,
    )
    if not len(idx):
        return
    top = view[y_cols].max(axis=1).to_numpy()[idx]
    extra_mat = None
    use_extra_cols = None
    if extra_cols and extra_series:
        cols_ok = [c for c in extra_cols if c in extra_series]
        if cols_ok:
            use_extra_cols = cols_ok
            rows_idx = view.index.to_numpy()[idx]
            extra_mat = np.column_stack(
                [
                    np.asarray(extra_series[c].reindex(rows_idx), dtype=np.float64)
                    for c in cols_ok
                ]
            )
    texts = _ranked_hover_texts_matrix(
        view,
        cols,
        idx,
        top_n=top_n,
        extra_cols=use_extra_cols,
        extra_mat=extra_mat,
    )
    fig.add_trace(
        go.Scattergl(
            x=to_plot_times(view["time"].to_numpy()[idx]),
            y=top,
            mode="markers",
            marker=dict(size=14, opacity=0),
            name=uid,
            showlegend=False,
            uid=uid,
            text=texts,
            hovertemplate="%{x|%Y년 %m월 %d일 %H:%M}<br>%{text}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def ranked_hover_html(
    ts,
    row: pd.Series,
    cols: list[str],
    *,
    cols_per_row: int = 4,
    extra_pairs: list[tuple[str, float]] | None = None,
) -> str:
    """Scrollable grid of metric values, largest first, left-to-right then wrap."""
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    if extra_pairs:
        pairs = pairs + list(extra_pairs)
        pairs.sort(key=lambda x: x[1], reverse=True)
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
                val=format_metric_value(
                    val, metric=name, attempt=_row_attempt(row)
                ),
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
    times = to_plot_times(df["time"])
    out: dict[str, tuple[Any, Any]] = {}
    for col in cols:
        values = df[col].to_numpy()
        if include_context:
            detail_start, detail_end = detail_window(df, start, end, pad_ratio=pad_ratio)
            pos = window_positions(df, detail_start, detail_end)
            idx = window_indices(values, pos, max_points)
        else:
            idx = window_plot_indices(values, df, start, end, max_points)
        out[col] = (times[idx], values[idx])
    return out


def add_label_indicator(
    fig: go.Figure, item: dict[str, Any], *, highlight: bool = False
) -> bool:
    """Draw a label overlay; return True for a range label.

    Unselected = yellow fill + edge lines above traces;
    selected = crimson fill/edges (also above).
    """
    s_utc = pd.to_datetime(item["start"], utc=True)
    e_utc = pd.to_datetime(item["end"], utc=True)
    s = to_plot_time(s_utc)
    e = to_plot_time(e_utc)
    is_range = item.get("kind") == "range" or s_utc != e_utc

    if highlight:
        color = "rgba(220, 20, 60, 0.42)"
        line = "crimson"
        line_w = 3
    else:
        color = ANOMALY_FILL
        line = ANOMALY_LINE
        line_w = 2

    def _edge(x, which: str, label_id: str) -> None:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(color=line, width=line_w),
            layer="above",
            editable=False,
            name=f"label_edge_{which}:{label_id}" if label_id else None,
        )

    if is_range:
        lid = str(item.get("id") or "")
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=s,
            x1=e,
            y0=0,
            y1=1,
            fillcolor=color,
            line=dict(width=0),
            layer="above",
            editable=False,
            name=f"label_fill:{lid}" if lid else None,
        )
        _edge(s, "start", lid)
        _edge(e, "end", lid)
    else:
        lid = str(item.get("id") or "")
        _edge(s, "start", lid)
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
    reason = label_reason_summary(item)
    if kind == "point":
        text = f"[점] anomaly · {start} · {reason}"
    else:
        text = f"[구간] anomaly · {start} → {end} · {reason}"
    return f"{text}  ({item.get('id')})"


def pred_line(
    item: dict[str, Any],
    *,
    feature_mode: str | None = None,
    with_contrib: bool = True,
) -> str:
    """One-line description of a model prediction (with top contributing metrics)."""
    kind = (item.get("kind") or "point").lower()
    start = format_kst(item.get("start"))
    end = format_kst(item.get("end"))
    if kind == "point":
        text = f"[점] model · {start}"
    else:
        text = f"[구간] model · {start} → {end}"
    if with_contrib:
        tops = item.get("top_metrics") or []
        if tops:
            names = ", ".join(
                display_training_feature(str(t.get("metric")), feature_mode=feature_mode)
                for t in tops[:3]
                if t.get("metric")
            )
            if names:
                text = f"{text} · 기여 {names}"
    return f"{text}  ({item.get('id')})"


def predictions_to_frame(doc: dict[str, Any] | None) -> pd.DataFrame:
    """Table of model prediction segments for the labeling UI."""
    rows = []
    for item in (doc or {}).get("labels") or []:
        tops = item.get("top_metrics") or []
        top_str = ", ".join(
            f"{display_metric(str(t.get('metric')))}"
            f"({float(t.get('contribution', 0)):.1f})"
            for t in tops[:5]
            if t.get("metric")
        )
        kind = item.get("kind", "point")
        start = format_kst(item.get("start"))
        end = format_kst(item.get("end"))
        interval = start if kind == "point" else f"{start} → {end}"
        rows.append(
            {
                "id": item.get("id"),
                "형식": "점" if kind == "point" else "구간",
                "시각 / 구간 (KST)": interval,
                "기여 metric (상위)": top_str,
                "score": item.get("score"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "형식",
                "시각 / 구간 (KST)",
                "기여 metric (상위)",
                "score",
            ]
        )
    return pd.DataFrame(rows)


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


def value_cursor_overlays(ts) -> dict[str, Any]:
    """Vertical cursor for keyboard/click value inspection (no label — bottom panel)."""
    x = to_plot_time(ts)
    return dict(
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
        fillcolor=ANOMALY_FILL if show else "rgba(0,0,0,0)",
        line=dict(width=0),
        layer="above",
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
    highlight_id: str | None = None,
    highlight_pred_id: str | None = None,
    predictions: dict[str, Any] | None = None,
    show_predictions: bool = False,
) -> go.Figure:
    """Trend chart: optional anomaly-score row, rates, then count metrics."""
    view = _filter_df(df, start, end) if filter_data else df
    # `metrics=[]` must stay empty — do not treat it as falsy fallback to all cols.
    cols = metric_columns(df) if metrics is None else list(metrics)
    if max_metrics is not None and max_metrics >= 0:
        cols = cols[:max_metrics]
    plot_cols = list(cols)
    primary_cols = [c for c in plot_cols if not is_rate_metric(c)]
    available_rates = rate_metrics_available(df)
    if metrics is None:
        rate_cols = available_rates
    else:
        selected = set(cols)
        rate_cols = [c for c in available_rates if c in selected]
    color_source = list(color_metrics) if color_metrics else list(cols)

    series_cols = list(dict.fromkeys([*primary_cols, *rate_cols]))
    has_score_col = bool(
        show_predictions and ANOMALY_SCORE_KEY in view.columns and len(view)
    )
    if has_score_col:
        series_cols = list(dict.fromkeys([*series_cols, ANOMALY_SCORE_KEY]))
    if not len(view):
        series = {}
    elif filter_data:
        series = plot_series(view, series_cols, max_points)
    else:
        series = plot_series_window(
            view, series_cols, start, end, max_points, include_context=False
        )

    threshold = None
    if show_predictions and predictions:
        ss = predictions.get("score_series") or {}
        if ss.get("threshold") is not None:
            threshold = float(ss["threshold"])
        elif predictions.get("threshold") is not None:
            threshold = float(predictions["threshold"])

    if has_score_col:
        score_x, score_y = series.get(ANOMALY_SCORE_KEY, ([], []))
        show_score = bool(len(score_y) and np.any(np.isfinite(np.asarray(score_y, dtype=float))))
    else:
        # Fallback when df was not warmed with ensure_score_column.
        score_vals_all = (
            align_scores_to_df(view, predictions)
            if show_predictions and predictions
            else np.full(0, np.nan, dtype=np.float64)
        )
        show_score = bool(len(score_vals_all) and np.any(np.isfinite(score_vals_all)))
        score_x, score_y = [], []
        if show_score and len(view):
            idx = window_plot_indices(score_vals_all, view, start, end, max_points)
            scored_pos = np.flatnonzero(np.isfinite(score_vals_all))
            if len(scored_pos):
                if start is not None and end is not None:
                    t0 = pd.to_datetime(start, utc=True)
                    t1 = pd.to_datetime(end, utc=True)
                    times_ns = _df_time_ns(view)
                    t0n, t1n = int(t0.value), int(t1.value)
                    in_win = scored_pos[
                        (times_ns[scored_pos] >= t0n) & (times_ns[scored_pos] <= t1n)
                    ]
                    pool = in_win if len(in_win) else scored_pos
                else:
                    pool = scored_pos
                if len(pool) > max_points:
                    step = max(1, len(pool) // max_points)
                    pool = pool[::step]
                idx = pool
            times_plot = to_plot_times(view["time"])
            score_x, score_y = times_plot[idx], score_vals_all[idx]

    rate_colors = {S_RATE_KEY: S_RATE_COLOR, A_RATE_KEY: A_RATE_COLOR}
    rate_hover = "값=%{y:.1%}<extra></extra>"
    count_hover = "값=%{y:,.0f}<extra></extra>"
    # Ranked tooltips (heavy) only while zoomed-in. Full / near-full windows
    # keep normal per-series hover so 「전체」 rebuilds stay light.
    embed_ranked = bool(hover_values) and _should_embed_ranked_hover(df, start, end)
    series_hoverinfo = "skip" if embed_ranked else None

    if show_score:
        n_rows = 3
        row_heights = [0.22, 0.28, 0.50]
        titles = ("Anomaly score", "SUCCESS / COMB rate", "Metric 값")
        rate_row, metric_row = 2, 3
        spacing = 0.08
    else:
        n_rows = 2
        row_heights = [0.5, 0.5]
        titles = ("SUCCESS / COMB rate", "Metric 값")
        rate_row, metric_row = 1, 2
        spacing = 0.13

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=spacing,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    if show_score and len(score_x):
        fig.add_trace(
            go.Scattergl(
                x=score_x,
                y=score_y,
                mode="lines",
                name="Anomaly score",
                opacity=0.9,
                line=dict(color=ANOMALY_SCORE_COLOR, width=1.5),
                # Custom place-time tip owns hover text; keep pickable for 값 탐색 clicks.
                hoverinfo="none",
                uid=ANOMALY_SCORE_KEY,
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line=dict(
                    color=ANOMALY_SCORE_THRESHOLD_COLOR,
                    width=1.5,
                    dash="dot",
                ),
                annotation_text="threshold",
                annotation_position="top left",
                annotation_font=dict(size=10, color=ANOMALY_SCORE_THRESHOLD_COLOR),
                row=1,
                col=1,
            )

    for col in rate_cols:
        x, y = series.get(col, ([], []))
        metric_name = display_metric(col)
        rate_kw: dict[str, Any] = dict(
            x=x,
            y=y,
            mode="lines",
            name=metric_name,
            opacity=0.72,
            line=dict(
                color=rate_colors.get(col, "#888"),
                width=1,
            ),
            uid=col,
        )
        if series_hoverinfo:
            rate_kw["hoverinfo"] = series_hoverinfo
        else:
            rate_kw["hovertemplate"] = (
                f"<b>{metric_name}</b><br>"
                "%{x|%Y년 %m월 %d일 %H:%M}<br>"
                f"{rate_hover}"
            )
        fig.add_trace(go.Scattergl(**rate_kw), row=rate_row, col=1)

    for i, col in enumerate(primary_cols):
        x, y = series.get(col, ([], []))
        metric_name = display_metric(col)
        try:
            color_i = color_source.index(col)
        except ValueError:
            color_i = i
        color = SERIES_COLORWAY[color_i % len(SERIES_COLORWAY)]
        metric_kw: dict[str, Any] = dict(
            x=x,
            y=y,
            mode="lines",
            name=metric_name,
            opacity=0.72,
            line=dict(width=1, color=color),
            uid=col,
        )
        if series_hoverinfo:
            metric_kw["hoverinfo"] = series_hoverinfo
        else:
            metric_kw["hovertemplate"] = (
                f"<b>{metric_name}</b><br>"
                "%{x|%Y년 %m월 %d일 %H:%M}<br>"
                f"{count_hover}"
            )
        fig.add_trace(go.Scattergl(**metric_kw), row=metric_row, col=1)

    data_rev = str(doc.get("plmn") or "labeling")
    if start is not None and end is not None:
        data_rev = (
            f"{data_rev}:{pd.to_datetime(start, utc=True).value}:"
            f"{pd.to_datetime(end, utc=True).value}"
        )
    if show_score:
        data_rev = f"{data_rev}:score"

    if embed_ranked and len(view):
        # Rate panel: SUCC / COMB largest-first (full max_points sample density).
        if rate_cols:
            _add_ranked_hover_anchor(
                fig,
                view,
                cols=list(rate_cols),
                row=rate_row,
                uid="rate-values",
                start=start,
                end=end,
                max_points=max_points,
                filter_data=filter_data,
            )

        # Metric panel: count metrics largest-first.
        metric_hover_cols = list(primary_cols)
        if metric_hover_cols:
            _add_ranked_hover_anchor(
                fig,
                view,
                cols=metric_hover_cols,
                row=metric_row,
                uid="metric-values",
                start=start,
                end=end,
                max_points=max_points,
                filter_data=filter_data,
            )
    elif len(view) and (primary_cols or rate_cols) and not hover_values:
        # Keep a non-hover hit target for click/inspect when ranked tips are off.
        hover_y_cols = primary_cols or rate_cols
        hover_row = metric_row if primary_cols else rate_row
        idx = _hover_sample_indices(
            view,
            hover_y_cols,
            start=start,
            end=end,
            max_points=max_points,
            filter_data=filter_data,
        )
        fig.add_trace(
            go.Scattergl(
                x=to_plot_times(view["time"].to_numpy()[idx]),
                y=view[hover_y_cols].max(axis=1).to_numpy()[idx],
                mode="markers",
                marker=dict(size=10, opacity=0),
                name="values",
                showlegend=False,
                uid="values",
                hoverinfo="skip",
            ),
            row=hover_row,
            col=1,
        )

    y_ref = view[primary_cols].max(axis=1) if len(view) and primary_cols else None

    if show_labels:
        hi = str(highlight_id) if highlight_id is not None else None
        for item in doc.get("labels", []):
            s_utc = pd.to_datetime(item["start"], utc=True)
            label_id = item.get("id", "")
            selected = hi is not None and str(label_id) == hi
            line = "crimson" if selected else ANOMALY_LINE
            add_label_indicator(fig, item, highlight=selected)

            if y_ref is not None and len(view):
                nearest = (view["time"] - s_utc).abs().idxmin()
                fig.add_trace(
                    go.Scatter(
                        x=[to_plot_time(view.loc[nearest, "time"])],
                        y=[float(y_ref.loc[nearest])],
                        mode="markers",
                        marker=dict(
                            size=14 if selected else 10,
                            color=line,
                            symbol="x",
                        ),
                        name=f"anomaly:{label_id}",
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=metric_row,
                    col=1,
                )

    if show_predictions and predictions:
        hid_pred = str(highlight_pred_id) if highlight_pred_id is not None else None
        overlay_items = model_overlays_for_plot(
            predictions.get("labels") or [],
            start=start,
            end=end,
            highlight_id=hid_pred,
        )
        # Edges only for the selection (or when few bands) — hundreds of
        # edge shapes freeze Plotly when Anomaly Transformer fires many segs.
        draw_all_edges = len(overlay_items) <= 16
        for item in overlay_items:
            is_hi = bool(hid_pred and str(item.get("id") or "") == hid_pred)
            add_model_indicator(
                fig,
                item,
                highlight=is_hi,
                draw_edges=draw_all_edges or is_hi,
            )

    fig.update_layout(
        title=title or f"{display_plmn(str(doc.get('plmn')))} anomaly labeling",
        height=480 if show_score else 420,
        autosize=True,
        # Ranked hover uses shared-x tips; otherwise closest (one series).
        hovermode="x" if embed_ranked else "closest",
        dragmode="zoom",
        showlegend=False,
        margin=dict(l=52, r=24, t=60, b=40),
        uirevision=str(doc.get("plmn") or "labeling"),
        datarevision=data_rev,
        colorway=list(SERIES_COLORWAY),
        plot_bgcolor="#ddd8cf",
        paper_bgcolor="#f3f1ec",
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

    if show_score:
        fig.update_yaxes(
            title_text="score",
            autorange=True,
            fixedrange=True,
            showgrid=True,
            showline=True,
            mirror=True,
            linewidth=1.5,
            linecolor="#6d28d9",
            gridcolor="rgba(124,58,237,0.22)",
            zeroline=False,
            title_font=dict(size=11, color="#5b21b6"),
            tickfont=dict(size=10, color="#5b21b6"),
            row=1,
            col=1,
        )

    fig.update_yaxes(
        title_text="rate",
        autorange=True,
        fixedrange=True,
        showgrid=True,
        showline=True,
        mirror=True,
        linewidth=1.5,
        linecolor="#5f8499",
        gridcolor="rgba(95,132,153,0.28)",
        zeroline=False,
        title_font=dict(size=11, color="#3d6578"),
        tickfont=dict(size=10, color="#3d6578"),
        row=rate_row,
        col=1,
    )
    fig.update_yaxes(
        title_text="value",
        fixedrange=True,
        showline=True,
        mirror=True,
        linewidth=1.5,
        linecolor="#8a7f6e",
        gridcolor="rgba(138,127,110,0.28)",
        zeroline=False,
        title_font=dict(size=11, color="#5c5346"),
        tickfont=dict(size=10, color="#5c5346"),
        row=metric_row,
        col=1,
    )

    tmin, tmax = data_time_bounds(view if len(view) else df)
    x0 = pd.to_datetime(start, utc=True) if start is not None else tmin
    x1 = pd.to_datetime(end, utc=True) if end is not None else tmax
    x0, x1 = clamp_time_range(x0, x1, tmin, tmax)
    x_range = [to_plot_time(x0), to_plot_time(x1)]
    x_axis_kwargs = dict(
        range=x_range,
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
        showline=True,
        mirror=True,
        linewidth=1.5,
    )
    for row_i in range(1, n_rows + 1):
        kw = dict(x_axis_kwargs)
        if row_i == 1:
            kw["matches"] = None
            kw["linecolor"] = "#6d28d9" if show_score else "#5f8499"
            kw["tickfont"] = dict(
                size=10, color="#5b21b6" if show_score else "#3d6578"
            )
        elif row_i == n_rows:
            kw["matches"] = "x"
            kw["title_text"] = "시간 (KST)"
            kw["linecolor"] = "#8a7f6e"
            kw["tickfont"] = dict(size=10, color="#5c5346")
            kw["title_font"] = dict(size=11, color="#5c5346")
        else:
            kw["matches"] = "x"
            kw["linecolor"] = "#5f8499"
            kw["tickfont"] = dict(size=10, color="#3d6578")
            kw["showticklabels"] = False
        fig.update_xaxes(**kw, row=row_i, col=1)

    # Panel fills first, then train|valid|test tints (same layer=below → later wins).
    _style_panel_separation(fig, n_rows=n_rows, show_score=show_score)
    add_split_region_overlays(fig, chrono_split_bounds(df, predictions))
    return fig


def _style_panel_separation(
    fig: go.Figure,
    *,
    n_rows: int = 2,
    show_score: bool = False,
) -> None:
    """Distinct fills / frames / subtitle chips for subplot rows."""
    domains = []
    for i in range(1, n_rows + 1):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        axis = getattr(fig.layout, key, None)
        dom = getattr(axis, "domain", None) if axis is not None else None
        domains.append(dom)
    if n_rows == 3:
        defaults = [(0.72, 1.0), (0.38, 0.64), (0.0, 0.30)]
    else:
        defaults = [(0.565, 1.0), (0.0, 0.435)]
    for i, dom in enumerate(domains):
        if not dom:
            domains[i] = defaults[i] if i < len(defaults) else (0.0, 1.0)

    x_key = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    x_axis = getattr(fig.layout, x_key, None) or fig.layout.xaxis
    x_dom = getattr(x_axis, "domain", None) or (0.0, 1.0)

    styles = []
    if show_score and n_rows == 3:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#e7f1f7", "#5f8499", "panel_bg_rate"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
        ]
    else:
        styles = [
            ("#e7f1f7", "#5f8499", "panel_bg_rate"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
        ]

    for i, (fill, border, name) in enumerate(styles):
        if i >= len(domains):
            break
        y0, y1 = domains[i]
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_dom[0],
            x1=x_dom[1],
            y0=y0,
            y1=y1,
            fillcolor=fill,
            line=dict(color=border, width=1.5),
            layer="below",
            editable=False,
            name=name,
        )

    for i in range(len(domains) - 1):
        gap_mid = (domains[i + 1][1] + domains[i][0]) / 2
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.01,
            x1=0.99,
            y0=gap_mid,
            y1=gap_mid,
            line=dict(color="#9a9286", width=2),
            layer="above",
            editable=False,
            name=f"panel_divider_{i}",
        )

    title_styles = (
        dict(
            font=dict(size=12, color="#5b21b6", family="sans-serif"),
            bgcolor="rgba(239,231,255,0.96)",
            bordercolor="#6d28d9",
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            font=dict(size=12, color="#1f4f63", family="sans-serif"),
            bgcolor="rgba(231,241,247,0.96)",
            bordercolor="#5f8499",
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            font=dict(size=12, color="#4a4338", family="sans-serif"),
            bgcolor="rgba(247,245,241,0.96)",
            bordercolor="#8a7f6e",
            borderwidth=1,
            borderpad=4,
        ),
    )
    if not show_score:
        title_styles = title_styles[1:]
    anns = list(fig.layout.annotations or ())
    for i, style in enumerate(title_styles):
        if i >= len(anns):
            break
        data = anns[i].to_plotly_json() if hasattr(anns[i], "to_plotly_json") else dict(anns[i])
        data.update(style)
        anns[i] = data
    if anns:
        fig.layout.annotations = tuple(anns)


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
            hovertemplate="%{x|%Y년 %m월 %d일 %H:%M}<br>%{y:,.0f}<extra></extra>",
        )
    )
    for item in doc.get("labels", []):
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
