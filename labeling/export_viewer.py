#!/usr/bin/env python3
"""Export static JSON for the GitHub Pages viewer (two catalogs).

Catalogs (selected with ``--catalog``):

- **labeled** (default): PLMNs with non-empty ``labeling/labels/*_labels.json``
  → ``docs/viewer/data/`` · high resolution (≤ ~20 operators recommended)
- **top100**: every PLMN in repo-root ``top100.txt``
  → ``docs/viewer/data-top100/`` · downsampled (default 5k points)
- **both**: run labeled + top100 (use before ``git push`` for the full web UI)

Usage:
    python labeling/export_viewer.py                      # both (labeled + top100)
    python labeling/export_viewer.py --catalog labeled    # labeled only
    python labeling/export_viewer.py --catalog both       # Pages: 라벨링 + Top 100 탭
    python labeling/export_viewer.py --catalog top100   # Top 100 only
    python labeling/export_viewer.py --plmn P0480 P0193

Git / Pages:
    python labeling/export_viewer.py --catalog both
    git add docs/viewer/ && git commit && git push

Source ``labeling/labels/*.json`` can stay gitignored; labels are embedded in
labeled JSON when present. Top-100 JSON has no anomaly UI on the web tab.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tool import (  # noqa: E402
    LABEL_DIR,
    A_RATE_KEY,
    M971_COL,
    S_RATE_KEY,
    data_time_bounds,
    format_kst,
    is_rate_metric,
    label_line,
    load_labels,
    load_or_build_ranking,
    load_plmn,
    metric_columns,
    minmax_indices,
    m971_tod_mean_series,
    set_mapping_enabled,
    to_plot_times,
)

# Viewer JSON is public (GitHub Pages) — never embed PLMN/metric mapping.
set_mapping_enabled(False)

LABELED_OUT_DIR = os.path.join(os.path.dirname(ROOT), "docs", "viewer", "data")
TOP100_OUT_DIR = os.path.join(os.path.dirname(ROOT), "docs", "viewer", "data-top100")
VIEWER_DIR = os.path.join(os.path.dirname(ROOT), "docs", "viewer")
# Labeled-only set is small (≤~20): keep full 5-min cadence when possible.
DEFAULT_MAX_POINTS = 100_000
# Top-100 tab: downsample for local browsing (not committed to git by default).
DEFAULT_TOP100_MAX_POINTS = 5_000
TOP100_LIST_PATH = os.path.join(os.path.dirname(ROOT), "top100.txt")
DEFAULT_MAX_METRICS = 46
# Soft guidance — Pages stays light if labeling stays within this.
DEFAULT_LABELED_SOFT_CAP = 20
# Kept for CLI compatibility; densify-around-labels is disabled (see export_one).
DEFAULT_LABEL_PAD_MIN = 12 * 60


def stamp_viewer_assets(build_id: str | None = None) -> str:
    """Bump ?v= on app.js / style.css so browsers and GH CDN fetch fresh assets."""
    build_id = build_id or str(int(time.time()))
    index_html = os.path.join(VIEWER_DIR, "index.html")
    if not os.path.isfile(index_html):
        return build_id
    text = open(index_html, encoding="utf-8").read()
    text2 = re.sub(
        r'(href="style\.css)(?:\?v=[^"]*)?(")',
        rf"\1?v={build_id}\2",
        text,
    )
    text2 = re.sub(
        r'(src="app\.js)(?:\?v=[^"]*)?(")',
        rf"\1?v={build_id}\2",
        text2,
    )
    if text2 != text:
        with open(index_html, "w", encoding="utf-8") as f:
            f.write(text2)
        print(f"stamped viewer assets ?v={build_id}", flush=True)
    return build_id


def _round_series(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        if v != v:  # NaN
            out.append(None)
        else:
            out.append(round(float(v), 3))
    return out


def _top_plmns(top_n: int) -> list[str]:
    rank = load_or_build_ranking(top_n=None)
    return [str(p) for p in rank["PLMN"].tolist()[: int(top_n)]]


def _labeled_plmns() -> list[str]:
    """PLMNs that currently have a non-empty ``*_labels.json`` on disk."""
    if not os.path.isdir(LABEL_DIR):
        return []
    rank = load_or_build_ranking(top_n=None)
    order = {
        str(row.PLMN): int(row.rank)
        for row in rank.itertuples(index=False)
    }
    found: list[str] = []
    for name in os.listdir(LABEL_DIR):
        if not name.endswith("_labels.json"):
            continue
        plmn = name[: -len("_labels.json")]
        doc = load_labels(plmn)
        if not doc.get("labels"):
            continue
        found.append(plmn)
    found.sort(key=lambda p: order.get(p, 10**9))
    return found


def _top100_plmns() -> list[str]:
    """PLMNs listed in repo-root top100.txt (order preserved)."""
    if not os.path.isfile(TOP100_LIST_PATH):
        raise SystemExit(f"top100 list not found: {TOP100_LIST_PATH}")
    out: list[str] = []
    with open(TOP100_LIST_PATH, encoding="utf-8") as f:
        for line in f:
            plmn = line.strip()
            if plmn and not plmn.startswith("#"):
                out.append(plmn)
    return out


def _purge_stale_exports(keep_plmns: list[str], out_dir: str) -> None:
    """Remove viewer JSON not in the current export set."""
    if not os.path.isdir(out_dir):
        return
    keep = {f"{p}.json" for p in keep_plmns}
    keep.add("index.json")
    for name in os.listdir(out_dir):
        if not name.endswith(".json") or name in keep:
            continue
        path = os.path.join(out_dir, name)
        os.remove(path)
        print(f"removed stale {path}", flush=True)


def _merge_sample_indices(
    envelope: np.ndarray,
    *,
    max_points: int,
) -> np.ndarray:
    """Downsample: regular stride (keeps ~5-min cadence) + min/max extrema."""
    n = int(envelope.shape[0])
    if n <= max_points:
        return np.arange(n)
    # Prefer a regular time grid so zoom/inspect step near the native cadence.
    n_reg = max(2, int(max_points * 0.65))
    step = max(1, int(np.ceil(n / n_reg)))
    reg = np.arange(0, n, step, dtype=np.int64)
    n_mm = max(2, max_points - int(reg.shape[0]))
    mm = minmax_indices(envelope, n_mm)
    idx = np.unique(np.concatenate([reg, mm, [0, n - 1]]))
    idx = idx[idx < n]
    if idx.shape[0] > max_points:
        pick = np.linspace(0, idx.shape[0] - 1, max_points).astype(np.int64)
        idx = idx[pick]
    return idx


def export_one(
    plmn: str,
    *,
    max_points: int,
    max_metrics: int,
    label_pad_min: int,
) -> dict:
    df = load_plmn(plmn)
    rank_df = load_or_build_ranking(top_n=None)
    row = rank_df.loc[rank_df["PLMN"] == plmn]
    rank = int(row["rank"].iloc[0]) if len(row) else None
    doc = load_labels(plmn, rank=rank)
    label_items = list(doc.get("labels") or [])
    cols = [c for c in metric_columns(df) if not is_rate_metric(c)][:max_metrics]
    for rate in (S_RATE_KEY, A_RATE_KEY):
        if rate in df.columns and rate not in cols:
            cols.append(rate)
    if not cols:
        raise RuntimeError(f"no metrics for {plmn}")

    envelope = df[cols].max(axis=1).to_numpy(dtype="float64")
    # Uniform sampling only (label overlays stay full-fidelity as shapes).
    _ = label_pad_min  # CLI retained; no longer densifies the plotted series
    idx = _merge_sample_indices(envelope, max_points=max_points)
    times = to_plot_times(df["time"].to_numpy()[idx])
    t_str = [str(x)[:19].replace("T", " ") for x in times]

    metrics = {}
    for col in cols:
        metrics[str(col)] = _round_series(df[col].to_numpy()[idx])

    m971_tod_ref = None
    if M971_COL in cols:
        tod_series = m971_tod_mean_series(df)
        if tod_series is not None:
            m971_tod_ref = _round_series(tod_series.to_numpy()[idx])

    labels = []
    for item in label_items:
        labels.append(
            {
                "id": item.get("id"),
                "kind": item.get("kind", "point"),
                "tag": item.get("tag", "anomaly"),
                "start": item.get("start"),
                "end": item.get("end"),
                "line": label_line(item),
            }
        )

    tmin, tmax = data_time_bounds(df)
    payload = {
        "plmn": plmn,
        "display": plmn,
        "rank": rank,
        "n_points_raw": int(len(df)),
        "n_points": len(t_str),
        "n_points_dense": 0,
        "start_kst": format_kst(tmin),
        "end_kst": format_kst(tmax),
        "t": t_str,
        "metrics": metrics,
        "labels": labels,
    }
    if m971_tod_ref is not None:
        payload["m971_tod_ref"] = m971_tod_ref
    return payload


def export_catalog(
    plmns: list[str],
    out_dir: str,
    *,
    catalog: str,
    max_points: int,
    max_metrics: int,
    label_pad_min: int,
    top_n: int | None = None,
) -> None:
    if not plmns:
        raise SystemExit(f"No PLMNs to export for catalog={catalog!r}.")

    os.makedirs(out_dir, exist_ok=True)
    _purge_stale_exports(plmns, out_dir)
    index = []
    for i, plmn in enumerate(plmns, 1):
        print(f"[{catalog} {i}/{len(plmns)}] export {plmn} ...", flush=True)
        try:
            payload = export_one(
                plmn,
                max_points=max_points,
                max_metrics=max_metrics,
                label_pad_min=label_pad_min,
            )
        except Exception as exc:
            print(f"  !! skip {plmn}: {exc}", flush=True)
            continue
        path = os.path.join(out_dir, f"{plmn}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024
        print(
            f"  -> {path} ({size_kb:.0f} KiB, "
            f"{payload['n_points']} pts ({payload['n_points_dense']} dense), "
            f"{len(payload['metrics'])} metrics, "
            f"{len(payload['labels'])} labels)"
        )
        index.append(
            {
                "plmn": plmn,
                "display": payload["display"],
                "rank": payload["rank"],
                "n_labels": len(payload["labels"]),
                "start_kst": payload["start_kst"],
                "end_kst": payload["end_kst"],
                "file": f"{plmn}.json",
            }
        )

    if not index:
        raise SystemExit(f"No PLMNs exported successfully for catalog={catalog!r}.")

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "catalog": catalog,
                "build": None,
                "top_n": top_n,
                "plmns": index,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    total_mb = sum(
        os.path.getsize(os.path.join(out_dir, row["file"])) for row in index
    ) / (1024 * 1024)
    print(
        f"wrote {index_path} ({len(index)} PLMNs, data≈{total_mb:.1f} MiB)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plmn", nargs="*", default=None, help="PLMN ids to export")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Export top N by M971 rank (legacy; writes to labeled data/)",
    )
    parser.add_argument(
        "--catalog",
        choices=("labeled", "top100", "both"),
        default="both",
        help="labeled=data/, top100=data-top100/, both=both (default)",
    )
    parser.add_argument(
        "--all-labeled",
        action="store_true",
        help="Export every PLMN that currently has labels (default if --top-n/--plmn omitted)",
    )
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--top100-max-points",
        type=int,
        default=DEFAULT_TOP100_MAX_POINTS,
        help=f"Max points per PLMN for --catalog top100/both (default {DEFAULT_TOP100_MAX_POINTS})",
    )
    parser.add_argument("--max-metrics", type=int, default=DEFAULT_MAX_METRICS)
    parser.add_argument(
        "--label-pad-min",
        type=int,
        default=DEFAULT_LABEL_PAD_MIN,
        help="Full-resolution minutes kept on each side of a label",
    )
    args = parser.parse_args()

    labeled_max = args.max_points if args.max_points is not None else DEFAULT_MAX_POINTS
    build_id = stamp_viewer_assets()

    def labeled_plmns() -> list[str]:
        if args.plmn:
            return list(args.plmn)
        if args.top_n is not None:
            return _top_plmns(args.top_n)
        plmns = _labeled_plmns()
        if len(plmns) > DEFAULT_LABELED_SOFT_CAP:
            print(
                f"warning: {len(plmns)} labeled PLMNs "
                f"(soft cap {DEFAULT_LABELED_SOFT_CAP}); "
                f"Pages/git size may grow — keep labeling ≤{DEFAULT_LABELED_SOFT_CAP} "
                f"or pass --max-points lower",
                flush=True,
            )
        return plmns

    catalogs: list[tuple[str, str, list[str], int, int | None]] = []
    if args.catalog in ("labeled", "both"):
        catalogs.append(("labeled", LABELED_OUT_DIR, labeled_plmns(), labeled_max, args.top_n))
    if args.catalog in ("top100", "both"):
        catalogs.append(
            (
                "top100",
                TOP100_OUT_DIR,
                _top100_plmns(),
                int(args.top100_max_points),
                len(_top100_plmns()),
            )
        )

    for catalog, out_dir, plmns, max_points, top_n in catalogs:
        export_catalog(
            plmns,
            out_dir,
            catalog=catalog,
            max_points=max_points,
            max_metrics=args.max_metrics,
            label_pad_min=args.label_pad_min,
            top_n=top_n,
        )
        index_path = os.path.join(out_dir, "index.json")
        idx = json.loads(open(index_path, encoding="utf-8").read())
        idx["build"] = build_id
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)

    print(f"viewer assets ?v={build_id}", flush=True)


if __name__ == "__main__":
    main()
