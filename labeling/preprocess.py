#!/usr/bin/env python3
"""Rebuild labeling caches after data/*.csv changes.

Single pass over data/*.csv → private artifacts under labeling/ (gitignored):
  - labeling/cache/{PLMN}.pkl
  - labeling/labels/plmn_rank.csv
  - labeling/cache/plmn_rank.signature

Future data/*.csv rows are expected to cover only PLMNs listed in
repo-root top100.txt. By default this script caches that set (and prunes
other pickles).

Usage:
  # After adding/replacing files in data/
  python labeling/preprocess.py
  python labeling/preprocess.py --force          # wipe pickles, full rebuild
  python labeling/preprocess.py --all            # every PLMN seen in CSV
  python labeling/preprocess.py --plmn P0480

Without --force, pickles whose signature still matches the current data/
fingerprint are skipped. Label JSON files are never deleted or rewritten.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    save_ranking_cache,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_plmn_list(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _prune_caches(keep: set[str]) -> int:
    """Remove labeling/cache/*.pkl whose PLMN is not in keep."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    removed = 0
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".pkl"):
            continue
        plmn = name[:-4]
        if plmn in keep:
            continue
        os.remove(os.path.join(CACHE_DIR, name))
        removed += 1
    return removed


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
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under {os.path.abspath(DATA_DIR)}")

    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    m971_sum: dict[str, int] = defaultdict(int)

    print(f"scanning {len(csv_files)} CSV files...")
    t_scan = time.time()
    for i, file in enumerate(csv_files, start=1):
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "PLMN" not in df.columns:
            print(f"  skip {file} (no PLMN column)")
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
    save_ranking_cache(rank_df, signature)

    write_plmns = sorted(parts.keys())
    print(f"writing {len(write_plmns)} pickle caches (signature={signature})...")
    t_write = time.time()
    written = 0
    skipped = 0
    for i, plmn in enumerate(write_plmns, start=1):
        cache_path = os.path.join(CACHE_DIR, f"{plmn}.pkl")
        if not force and os.path.exists(cache_path):
            try:
                cached = pd.read_pickle(cache_path)
                if cached.attrs.get("signature") == signature:
                    skipped += 1
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
        parts[plmn] = []
        if i % 50 == 0 or i == len(write_plmns):
            print(f"  [{i}/{len(write_plmns)}] wrote {plmn} ({len(df):,} rows)")

    print(
        f"write done in {time.time() - t_write:.1f}s "
        f"(written={written}, skipped={skipped})"
    )
    return rank_df, written


def _default_top100_path() -> Path:
    return ROOT / "top100.txt"


def preprocess(
    *,
    top: int | None,
    plmns: list[str] | None,
    from_file: str | None,
    all_plmns: bool,
    force: bool,
    prune: bool,
) -> None:
    t0 = time.time()
    if force:
        n = clear_plmn_cache()
        print(f"cleared {n} cached pickle files")

    targets: set[str] | None = None
    if plmns:
        targets = set(plmns)
    elif all_plmns:
        targets = None
        print("caching every PLMN found in data/")
    elif from_file:
        path = Path(from_file)
        if not path.is_file():
            alt = ROOT / from_file
            path = alt if alt.is_file() else path
        targets = set(_read_plmn_list(path))
        print(f"from-file {path}: {len(targets)} PLMNs")
    elif top is not None:
        print("building ranking for --top filter...")
        from tool import _build_ranking_from_data

        rank_df = _build_ranking_from_data()
        save_ranking_cache(rank_df)
        targets = set(rank_df.head(top)["PLMN"].astype(str).tolist())
        print(f"filtering to top {top}: {len(targets)} PLMNs")
    else:
        # Default: top100.txt (future data is limited to this set)
        path = _default_top100_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Default PLMN list not found: {path}\n"
                "Pass --from-file, --top N, --plmn, or --all."
            )
        targets = set(_read_plmn_list(path))
        print(f"default {path.name}: {len(targets)} PLMNs")
        prune = True

    print(f"data dir:  {os.path.abspath(DATA_DIR)}")
    print(f"cache dir: {os.path.abspath(CACHE_DIR)}")
    print(f"labels dir:{os.path.abspath(LABEL_DIR)}")
    print("existing label JSON files: preserved")
    print(f"signature: {_data_signature(DATA_DIR)}")

    rank_df, written = _build_all_caches(targets=targets, force=force)

    if prune and targets is not None:
        removed = _prune_caches(targets)
        print(f"pruned {removed} pickle(s) outside selected set")

    print(f"rank file: {RANK_CACHE_PATH} ({len(rank_df)} PLMNs)")
    print(f"done in {time.time() - t0:.1f}s (written={written})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild PLMN caches after updating data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python labeling/preprocess.py\n"
            "  python labeling/preprocess.py --force\n"
            "  python labeling/preprocess.py --all\n"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only cache top-N PLMNs by M971 sum",
    )
    parser.add_argument(
        "--from-file",
        default=None,
        metavar="PATH",
        help="Only cache PLMN ids listed in a text file (one per line)",
    )
    parser.add_argument(
        "--plmn",
        action="append",
        default=None,
        help="Specific PLMN id (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Cache every PLMN found in data/ (ignore top100.txt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing pickles first, then rebuild",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="After rebuild, delete pickles not in the selected PLMN set "
        "(on by default when using top100.txt)",
    )
    args = parser.parse_args()
    preprocess(
        top=args.top,
        plmns=args.plmn,
        from_file=args.from_file,
        all_plmns=args.all,
        force=args.force,
        prune=args.prune,
    )


if __name__ == "__main__":
    main()
