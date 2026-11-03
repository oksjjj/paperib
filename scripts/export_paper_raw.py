#!/usr/bin/env python3
"""Filter IB raw CSVs down to paper PLMNs (de-identified IDs only).

Reads from ``data/ib_data/_local_raw_full/`` (or ``--src``) and writes
``data/ib_data/raw/masked_*.csv`` for git. Mapping files are never copied.

Example:
  .venv/bin/python scripts/export_paper_raw.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "labeling"))
from tool import PAPER_PLMNS  # noqa: E402

DEFAULT_SRC = REPO / "data" / "ib_data" / "_local_raw_full"
DEFAULT_DST = REPO / "data" / "ib_data" / "raw"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = p.parse_args()
    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        raise SystemExit(f"missing source dir: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    keep = set(PAPER_PLMNS)
    files = sorted(src.glob("masked_*.csv"))
    if not files:
        raise SystemExit(f"no masked_*.csv under {src}")
    total_in = total_out = 0
    for f in files:
        df = pd.read_csv(f)
        if "PLMN" not in df.columns:
            raise SystemExit(f"{f.name}: no PLMN column")
        sub = df[df["PLMN"].astype(str).isin(keep)]
        total_in += len(df)
        total_out += len(sub)
        out = dst / f.name
        sub.to_csv(out, index=False)
        print(f"{f.name}: {len(df)} → {len(sub)}")
    print(
        f"done: {len(files)} files, rows {total_in} → {total_out}, "
        f"PLMNs={len(keep)} → {dst}"
    )
    print("NOTE: do not commit *_mapping.txt (stay local-only).")


if __name__ == "__main__":
    main()
