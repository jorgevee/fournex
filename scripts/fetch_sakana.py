#!/usr/bin/env python3
"""Fetch a small, offline-friendly subset of the SakanaAI/AI-CUDA-Engineer-Archive.

Dependency-free: talks to the HuggingFace datasets-server REST ``/rows`` endpoint
with urllib (no ``datasets`` package, no pandas). Keeps only rows that carry a
usable ``NCU_Profile`` (the column is null on many rows) and whose large cells
were not truncated by the server, then writes one JSON object per line.

The result feeds ``frx eval sakana`` and the test suite, so it must run fully
offline once cached. Re-run this only to refresh or extend the cached subset.

Usage:
    python scripts/fetch_sakana.py --out backend/python/fournex/data/sakana/subset.jsonl \
        --per-level 30 --scan 200
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "SakanaAI/AI-CUDA-Engineer-Archive"
CONFIG = "default"
SPLITS = ("level_1", "level_2", "level_3")
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # datasets-server hard cap on `length`

# Columns we keep. Torch_Profile and PyTorch_Code_Functional are dropped to bound
# fixture size; everything the eval harness reads is retained.
KEEP_COLUMNS = (
    "Op_Name", "Level_ID", "Task_ID", "Kernel_Name",
    "CUDA_Runtime", "PyTorch_Native_Runtime", "PyTorch_Compile_Runtime",
    "CUDA_Speedup_Native", "CUDA_Speedup_Compile",
    "CUDA_Code", "PyTorch_Code_Module",
    "Correct", "Max_Diff", "Error",
    "NCU_Profile", "Clang_Tidy",
    "__index_level_0__",
)


def _fetch_page(split: str, offset: int, length: int) -> dict:
    params = urllib.parse.urlencode({
        "dataset": DATASET, "config": CONFIG, "split": split,
        "offset": offset, "length": length,
    })
    req = urllib.request.Request(f"{ROWS_URL}?{params}", headers={"User-Agent": "fournex-fetch"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ncu_is_usable(profile: object) -> bool:
    """A profile is usable if it parses as a Python-repr dict with a metrics block."""
    if not isinstance(profile, str) or not profile.strip():
        return False
    try:
        parsed = ast.literal_eval(profile)
    except (ValueError, SyntaxError):
        return False
    return isinstance(parsed, dict) and bool(parsed.get("metrics"))


def fetch_split(split: str, *, want: int, want_incorrect: int, scan: int) -> list[dict]:
    """Collect ``want`` correct rows that carry a usable NCU profile, plus up to
    ``want_incorrect`` incorrect rows (NCU optional — compile failures have none).

    Incorrect kernels are scarce in the early rows, so we keep scanning for them
    even after the correct quota is full. They anchor the correctness-warning
    evaluation, which never relies on the NCU profile.
    """
    correct: list[dict] = []
    incorrect: list[dict] = []
    offset = 0
    while offset < scan and (len(correct) < want or len(incorrect) < want_incorrect):
        length = min(PAGE, scan - offset)
        payload = _fetch_page(split, offset, length)
        entries = payload.get("rows", [])
        if not entries:
            break
        for entry in entries:
            row = entry.get("row", {})
            truncated = {c for c in entry.get("truncated_cells", [])}
            if "CUDA_Code" in truncated:
                continue
            kept_row = {k: row.get(k) for k in KEEP_COLUMNS}
            if not row.get("Correct", True):
                if len(incorrect) < want_incorrect and "NCU_Profile" not in truncated:
                    incorrect.append(kept_row)
            elif len(correct) < want:
                if "NCU_Profile" in truncated or not _ncu_is_usable(row.get("NCU_Profile")):
                    continue
                correct.append(kept_row)
        offset += length
        time.sleep(0.2)  # be polite to the public endpoint
    return correct + incorrect


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--per-level", type=int, default=30, help="correct rows w/ NCU to keep per level")
    ap.add_argument("--incorrect-per-level", type=int, default=8, help="incorrect rows to keep per level")
    ap.add_argument("--scan", type=int, default=300, help="max rows to scan per level")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for split in SPLITS:
        rows = fetch_split(
            split,
            want=args.per_level,
            want_incorrect=args.incorrect_per_level,
            scan=args.scan,
        )
        n_bad = sum(1 for r in rows if not r.get("Correct", True))
        print(f"{split}: kept {len(rows)} rows ({len(rows) - n_bad} correct+NCU, {n_bad} incorrect)", file=sys.stderr)
        all_rows.extend(rows)

    with out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_rows)} rows -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
