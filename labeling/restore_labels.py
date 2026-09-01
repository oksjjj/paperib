#!/usr/bin/env python3
"""Restore data/labels/*_labels.json from exported viewer JSON.

Viewer export embeds labels (id/kind/tag/start/end) in docs/viewer/data/*.json.
Use when local label files were lost but a prior export still exists in git.

Usage:
    python labeling/restore_labels.py
    python labeling/restore_labels.py --source ../docs/viewer/data
    python labeling/restore_labels.py --plmn P0480 P1888
    python labeling/restore_labels.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tool import LABEL_DIR, save_labels  # noqa: E402

DEFAULT_SOURCE = os.path.join(os.path.dirname(ROOT), "docs", "viewer", "data")


def _export_label_to_local(item: dict) -> dict:
    return {
        "id": item["id"],
        "kind": item.get("kind", "point"),
        "tag": item.get("tag", "anomaly"),
        "start": item["start"],
        "end": item.get("end") or item["start"],
        "metrics": item.get("metrics") or ["ALL"],
        "updated_at": item.get("updated_at"),
    }


def restore_from_export(
    source_dir: str,
    *,
    plmns: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> list[str]:
    index_path = os.path.join(source_dir, "index.json")
    if not os.path.isfile(index_path):
        raise SystemExit(f"index not found: {index_path}")

    idx = json.loads(open(index_path, encoding="utf-8").read())
    entries = idx.get("plmns") or []
    if plmns:
        want = set(plmns)
        entries = [row for row in entries if row.get("plmn") in want]

    restored: list[str] = []
    for row in entries:
        plmn = str(row.get("plmn") or "")
        if not plmn:
            continue
        export_path = os.path.join(source_dir, row.get("file") or f"{plmn}.json")
        if not os.path.isfile(export_path):
            print(f"skip {plmn}: missing {export_path}", flush=True)
            continue

        payload = json.loads(open(export_path, encoding="utf-8").read())
        labels = payload.get("labels") or []
        if not labels:
            print(f"skip {plmn}: no labels in export", flush=True)
            continue

        out_path = os.path.join(LABEL_DIR, f"{plmn}_labels.json")
        if os.path.isfile(out_path) and not force:
            print(f"skip {plmn}: {out_path} exists (use --force)", flush=True)
            continue

        doc = {
            "plmn": plmn,
            "rank": payload.get("rank", row.get("rank")),
            "labels": [_export_label_to_local(x) for x in labels],
        }
        if dry_run:
            print(
                f"would restore {plmn}: {len(doc['labels'])} labels -> {out_path}",
                flush=True,
            )
        else:
            path = save_labels(doc)
            print(f"restored {plmn}: {len(doc['labels'])} labels -> {path}", flush=True)
        restored.append(plmn)

    if not restored and not dry_run:
        print("nothing restored", flush=True)
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"viewer export dir with index.json (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--plmn", nargs="*", default=None, help="restore only these PLMNs")
    parser.add_argument("--dry-run", action="store_true", help="print actions only")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing data/labels/*_labels.json",
    )
    args = parser.parse_args()
    restore_from_export(
        os.path.abspath(args.source),
        plmns=args.plmn,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
