#!/usr/bin/env python3
"""raw CSV가 변경된 뒤 공통 IB 데이터 캐시와 PLMN 순위를 다시 만드는 스크립트.

data/ib_data/raw 아래의 CSV를 한 번씩 읽어 다음 산출물을 만든다.

- data/ib_data/cache/{PLMN}.pkl
- data/ib_data/labels/plmn_rank.csv
- data/ib_data/cache/plmn_rank.signature

기본 실행은 저장소 루트의 top100.txt에 포함된 사업자만 캐시하고
목록 밖의 오래된 pickle은 정리한다.

사용 예:
  python scripts/build_ui_cache.py
  python scripts/build_ui_cache.py --force
  python scripts/build_ui_cache.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    LABEL_DIR,
    RANK_CACHE_PATH,
    build_plmn_ranking,
    clear_plmn_cache,
    data_signature,
    data_signature_text,
    save_ranking_cache,
)


def _read_plmn_list(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _prune_caches(keep: set[str]) -> int:
    """keep 목록에 없는 사업자의 공통 PLMN cache를 삭제하고 삭제 개수를 반환한다."""
    if not CACHE_DIR.is_dir():
        return 0
    removed = 0
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".pkl") or "_score_" in name:
            continue
        plmn = name[:-4]
        if plmn in keep:
            continue
        os.remove(str(CACHE_DIR / name))
        removed += 1
    return removed


def _build_all_caches(
    *,
    targets: set[str] | None,
    force: bool,
) -> tuple[pd.DataFrame, int]:
    """모든 CSV를 한 번씩만 읽어 사업자별 pickle을 만들고 트래픽 순위 DataFrame을 반환한다.

    파일마다 모든 사업자를 반복해서 다시 읽는 대신, 한 번의 scan에서 PLMN별 조각을
    모아두었다가 마지막에 합치므로 대량 CSV에서도 불필요한 I/O를 줄인다.
    """
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    signature = data_signature(DATA_DIR)

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under {str(DATA_DIR.resolve())}")

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

    # 선택된 캐시 대상과 관계없이 M971 합계를 확인할 수 있는 모든 PLMN의 전체 순위는 항상 저장한다.
    rank_df = (
        pd.DataFrame({"PLMN": list(m971_sum.keys()), "M971_sum": list(m971_sum.values())})
        .sort_values("M971_sum", ascending=False)
        .reset_index(drop=True)
    )
    rank_df.insert(0, "rank", rank_df.index + 1)
    save_ranking_cache(rank_df, data_signature_text(DATA_DIR))

    write_plmns = sorted(parts.keys())
    print(f"writing {len(write_plmns)} pickle caches (signature={signature})...")
    t_write = time.time()
    written = 0
    skipped = 0
    for i, plmn in enumerate(write_plmns, start=1):
        cache_path = str(CACHE_DIR / f"{plmn}.pkl")
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
        rank_df = build_plmn_ranking(DATA_DIR)
        save_ranking_cache(rank_df)
        targets = set(rank_df.head(top)["PLMN"].astype(str).tolist())
        print(f"filtering to top {top}: {len(targets)} PLMNs")
    else:
        # 별도 옵션이 없으면 top100.txt를 기준으로 캐시 대상을 제한한다.
        path = _default_top100_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Default PLMN list not found: {path}\n"
                "Pass --from-file, --top N, --plmn, or --all."
            )
        targets = set(_read_plmn_list(path))
        print(f"default {path.name}: {len(targets)} PLMNs")
        prune = True

    print(f"data dir:  {str(DATA_DIR.resolve())}")
    print(f"cache dir: {str(CACHE_DIR.resolve())}")
    print(f"labels dir:{str(LABEL_DIR.resolve())}")
    print("existing label JSON files: preserved")
    print(f"signature: {data_signature_text(DATA_DIR)}")

    rank_df, written = _build_all_caches(targets=targets, force=force)

    if prune and targets is not None:
        removed = _prune_caches(targets)
        print(f"pruned {removed} pickle(s) outside selected set")

    print(f"rank file: {RANK_CACHE_PATH} ({len(rank_df)} PLMNs)")
    print(f"done in {time.time() - t0:.1f}s (written={written})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild shared IB PLMN caches after updating raw CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/build_ui_cache.py\n"
            "  python scripts/build_ui_cache.py --force\n"
            "  python scripts/build_ui_cache.py --all\n"
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
