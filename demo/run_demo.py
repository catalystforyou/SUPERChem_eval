#!/usr/bin/env python3
"""Run a no-API SUPERChem demo: accuracy on bundled Gemini 2.5 Pro answers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ANSWERS_FILE = (
    DEMO_DIR
    / "20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl"
)
SPLIT_MAP_FILE = DEMO_DIR / "dataset_split_map.json"
BASELINE_FILE = DEMO_DIR / "20251015_baseline_demo.csv"


def load_split_uuids() -> set[str]:
    with open(SPLIT_MAP_FILE, encoding="utf-8") as f:
        split_map = json.load(f)
    return {
        uuid
        for uuid, meta in split_map.items()
        if meta.get("split") in ("release", "holdout")
    }


def load_answers(valid_uuids: set[str]) -> dict[str, list[int]]:
    scores: dict[str, list[int]] = {}
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            uuid = record["uuid"]
            if uuid not in valid_uuids:
                continue
            scores.setdefault(uuid, []).append(int(record.get("score", 0)))
    return scores


def human_baseline_accuracy(valid_uuids: set[str]) -> tuple[float, int, int] | None:
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed; skipping human baseline.", file=sys.stderr)
        return None

    df = pd.read_csv(BASELINE_FILE)
    df = df[df["uuid"].isin(valid_uuids)]
    if df.empty:
        return None
    correct_by_uuid = df.groupby("uuid")["score"].max()
    num_correct = int(correct_by_uuid.sum())
    total = len(correct_by_uuid)
    return 100.0 * num_correct / total, num_correct, total


def main() -> int:
    print("SUPERChem demo — Gemini 2.5 Pro (text-only, high reasoning)")
    print(f"Data directory: {DEMO_DIR}\n")

    for path in (ANSWERS_FILE, SPLIT_MAP_FILE, DEMO_DIR / "questions_demo.parquet"):
        if not path.exists():
            print(f"Error: missing required file: {path}", file=sys.stderr)
            return 1

    valid_uuids = load_split_uuids()
    scores = load_answers(valid_uuids)
    if not scores:
        print("Error: no answer records matched the demo split map.", file=sys.stderr)
        return 1

    first_trial_correct = sum(1 for s in scores.values() if s and s[0] == 1)
    total = len(scores)
    accuracy = 100.0 * first_trial_correct / total

    print(f"Questions evaluated : {total}")
    print(f"Model               : gemini-2.5-pro (high, text-only)")
    print(f"Metric              : pass@1 (first trial)")
    print(f"Accuracy            : {accuracy:.1f}% ({first_trial_correct}/{total})")

    baseline = human_baseline_accuracy(valid_uuids)
    if baseline is not None:
        acc, correct, n = baseline
        print(f"Human baseline      : {acc:.1f}% ({correct}/{n}) on same demo items")

    print("\nPer-question results (uuid, score):")
    for uuid in sorted(scores):
        print(f"  {uuid}  score={scores[uuid][0]}")

    print(
        "\nDemo finished successfully. "
        "For full-benchmark evaluation and RPF/DAG scoring, see the root README."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
