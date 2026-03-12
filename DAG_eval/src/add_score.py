#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_scores(answer_path: Path) -> dict:
    scores = {}
    total = 0
    missing_uuid = 0
    dup = 0

    with answer_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uuid = obj.get("uuid")
            if not uuid:
                missing_uuid += 1
                continue
            if uuid in scores:
                dup += 1
            scores[uuid] = obj.get("score")

    print(
        f"[add_score] answers: total={total}, missing_uuid={missing_uuid}, dup_uuid={dup}",
        file=sys.stderr,
    )
    return scores


def add_score(answer_path: Path, judge_path: Path, output_path: Path) -> None:
    scores = load_scores(answer_path)
    total = 0
    matched = 0
    missing = 0
    invalid = 0

    with judge_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            uuid = obj.get("uuid")
            if uuid in scores:
                obj["score"] = scores[uuid]
                matched += 1
            else:
                obj["score"] = None
                missing += 1
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        "[add_score] judge: "
        f"total={total}, matched={matched}, missing={missing}, invalid={invalid}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add score from answer jsonl into judge jsonl by uuid."
    )
    parser.add_argument(
        "--answer",
        required=True,
        type=Path,
        help="Path to LLM answer jsonl file (contains uuid and score).",
    )
    parser.add_argument(
        "--judge",
        required=True,
        type=Path,
        help="Path to judge jsonl file (will receive score field).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path. Default: <judge>.with_score.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = args.judge.with_suffix(args.judge.suffix + ".jsonl")
    add_score(args.answer, args.judge, output_path)
    print(f"[add_score] wrote: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
