#!/usr/bin/env python3
"""Launch TensorBoard for a paperib OmniAnomaly PLMN run.

Examples:
    cd OmniAnomaly
    ../.venv/bin/python view_tensorboard.py --plmn P0480
    ../.venv/bin/python view_tensorboard.py --plmn P0480 --port 6007
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_logdir(plmn: str, run_name: str | None) -> str:
    base = os.path.join(ROOT, "log", plmn)
    if run_name:
        candidate = os.path.join(base, run_name)
        if os.path.isdir(candidate):
            return candidate
        raise SystemExit(f"log dir not found: {candidate}")
    if not os.path.isdir(base):
        raise SystemExit(
            f"no TensorBoard logs under {base}\n"
            f"Train first with: python run_plmn.py --plmn {plmn} ..."
        )
    # Prefer nested tensorboard/ if present; otherwise the PLMN log root.
    tb_hits = []
    for dirpath, dirnames, _ in os.walk(base):
        if os.path.basename(dirpath) == "tensorboard":
            tb_hits.append(dirpath)
        # prune heavy dirs
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
    if tb_hits:
        # Parent of stamp dirs so multiple runs show as side-by-side
        return os.path.commonpath(tb_hits) if len(tb_hits) > 1 else tb_hits[0]
    return base


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--run_name", default=None, help="Optional: paperib, …")
    p.add_argument("--port", type=int, default=6006)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    logdir = _find_logdir(args.plmn, args.run_name)
    # Has any event file?
    has_events = False
    for dirpath, _, filenames in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in filenames):
            has_events = True
            break
    if not has_events:
        raise SystemExit(
            f"no TensorBoard event files under {logdir}\n"
            f"Previous runs may have used --no_tensorboard, or training "
            f"has not finished. Re-train (TensorBoard is on by default):\n"
            f"  python run_plmn.py --plmn {args.plmn} --max_epoch 10 --stable_train"
        )

    tb = shutil.which("tensorboard")
    if tb is None:
        # venv console script next to this interpreter
        candidate = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        tb = candidate if os.path.isfile(candidate) else None
    if tb is None:
        raise SystemExit(
            "tensorboard executable not found; "
            "pip install -r requirements.txt"
        )

    cmd = [
        tb,
        "--logdir",
        logdir,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(f"logdir={logdir}", flush=True)
    print(f"open http://{args.host}:{args.port}/", flush=True)
    print(" ".join(cmd), flush=True)
    os.chdir(ROOT)
    os.execv(tb, cmd)


if __name__ == "__main__":
    main()
