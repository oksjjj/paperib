#!/usr/bin/env python3
"""Preprocess data for fast labeling loads.

Single-pass over data/*.csv → private artifacts under labeling/ (gitignored):
  - labeling/cache/{PLMN}.pkl
  - labeling/labels/plmn_rank.csv

Usage:
  python labeling/preprocess.py              # all PLMNs
  python labeling/preprocess.py --top 50
  python labeling/preprocess.py --plmn P0480
  python labeling/preprocess.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    LABEL_DIR,
    RANK_CACHE_PATH,
    _data_signature,
    clear_plmn_cache,
    ensure_label_dir,
)


def _build_all_caches(
    *,
    targets: set[str] | None,
    force: bool,
) -> tuple[pd.DataFrame, int]:
    """Scan every CSV once, write per-PLMN pickles, return ranking frame."""
    ensure_label_dir()
    os.makedirs(CACHE_DIR, exist_ok=True)
    signature = _data_signature(DATA_DIR)

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    m971_sum: dict[str, int] = defaultdict(int)

    print(f"scanning {len(csv_files)} CSV files...")
    t_scan = time.time()
    for i, file in enumerate(csv_files, start=1):
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path)
        if "PLMN" not in df.columns:
            continue
        if "M971" in df.columns:
            for plmn, value in df.groupby("PLMN")["M971"].sum().items():
                m971_sum[str(plmn)] += int(value)
        for plmn, group in df.groupby("PLMN", sort=False):
            plmn = str(plmn)
            if targets is not None and plmn not in targets:
                continue
            parts[plmn].append(group)
        if i % 10 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}] {file}  (groups={len(parts)})")
    print(f"scan done in {time.time() - t_scan:.1f}s")

    # Full ranking always written (all PLMNs seen in M971 sums)
    rank_df = (
        pd.DataFrame({"PLMN": list(m971_sum.keys()), "M971_sum": list(m971_sum.values())})
        .sort_values("M971_sum", ascending=False)
        .reset_index(drop=True)
    )
    rank_df.insert(0, "rank", rank_df.index + 1)
    rank_df.to_csv(RANK_CACHE_PATH, index=False)

    write_plmns = sorted(parts.keys())
    print(f"writing {len(write_plmns)} pickle caches...")
    t_write = time.time()
    written = 0
    for i, plmn in enumerate(write_plmns, start=1):
        cache_path = os.path.join(CACHE_DIR, f"{plmn}.pkl")
        if not force and os.path.exists(cache_path):
            # Keep existing if signature matches
            try:
                cached = pd.read_pickle(cache_path)
                if cached.attrs.get("signature") == signature:
                    if i % 100 == 0 or i == len(write_plmns):
                        print(f"  [{i}/{len(write_plmns)}] skip {plmn} (fresh)")
                    continue
            except Exception:
                pass

        df = pd.concat(parts[plmn], ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.attrs["signature"] = signature
        df.to_pickle(cache_path)
        written += 1
        # free memory
        parts[plmn] = []
        if i % 50 == 0 or i == len(write_plmns):
            print(f"  [{i}/{len(write_plmns)}] wrote {plmn} ({len(df):,} rows)")

    print(f"write done in {time.time() - t_write:.1f}s ({written} new/updated)")
    return rank_df, written


def preprocess(*, top: int | None, plmns: list[str] | None, force: bool) -> None:
    t0 = time.time()
    if force:
        n = clear_plmn_cache()
        print(f"cleared {n} cached files")

    targets: set[str] | None = None
    if plmns:
        targets = set(plmns)
    elif top is not None:
        # Need ranking first — quick M971-only pass, then cache top-N only
        print("building ranking for --top filter...")
        from tool import _build_ranking_from_data

        rank_df = _build_ranking_from_data()
        rank_df.to_csv(RANK_CACHE_PATH, index=False)
        targets = set(rank_df.head(top)["PLMN"].tolist())
        print(f"filtering to top {top}: {len(targets)} PLMNs")

    print(f"data dir: {os.path.abspath(DATA_DIR)}")
    print(f"cache dir: {os.path.abspath(CACHE_DIR)}")
    print(f"labels dir: {os.path.abspath(LABEL_DIR)}")
    print(f"signature: {_data_signature(DATA_DIR)}")

    rank_df, written = _build_all_caches(targets=targets, force=force)
    print(f"rank file: {RANK_CACHE_PATH} ({len(rank_df)} PLMNs)")
    print(f"done in {time.time() - t0:.1f}s (written={written})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PLMN caches for labeling")
    parser.add_argument("--top", type=int, default=None, help="Only top-N PLMNs by M971")
    parser.add_argument("--plmn", action="append", default=None, help="Specific PLMN (repeatable)")
    parser.add_argument("--force", action="store_true", help="Rebuild caches from CSV")
    args = parser.parse_args()
    preprocess(top=args.top, plmns=args.plmn, force=args.force)


if __name__ == "__main__":
    main()
