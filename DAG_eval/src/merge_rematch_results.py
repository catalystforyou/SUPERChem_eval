#!/usr/bin/env python3
"""
Merge rematch results script:
- Read match_results_tagged_output.jsonl (round0)
- Read all match_results_rematch_round*.jsonl
- Merge and deduplicate, preferring later round results
- Output only contains uuid, parsed, round fields
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge rematch results with deduplication (prefer later rounds)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing match_results files (e.g., gpt-5_minimal__false__model-v5/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output JSONL file. If not specified, will use <input_dir>/match_results_merged.jsonl",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def extract_round_number(filename: str) -> int:
    """Extract round number from filename"""
    match = re.search(r"_round(\d+)\.jsonl$", filename)
    if match:
        return int(match.group(1))
    return 0


def load_file(filepath: str, round_num: int) -> List[Dict]:
    """Load JSONL file and add round field"""
    records = []
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return records
    
    logger.info(f"Loading {filepath} (round {round_num})")
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                uuid = record.get("uuid")
                parsed = record.get("parsed")
                
                if not uuid:
                    logger.warning(f"Record missing uuid: {line[:100]}")
                    continue
                if not parsed:
                    logger.warning(f"Record {uuid} missing parsed field")
                    continue
                
                records.append({
                    "uuid": uuid,
                    "parsed": parsed,
                    "round": round_num,
                })
                count += 1
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                continue
    
    logger.info(f"Loaded {count} records from round {round_num}")
    return records


def find_files(input_dir: str) -> Tuple[str, List[Tuple[str, int]]]:
    """Find match_results_tagged_output.jsonl and all rematch files"""
    base_file = os.path.join(input_dir, "match_results_tagged_output.jsonl")
    
    # Find all rematch files
    rematch_files = []
    for filename in os.listdir(input_dir):
        if filename.startswith("match_results_rematch_round") and filename.endswith(".jsonl"):
            filepath = os.path.join(input_dir, filename)
            round_num = extract_round_number(filename)
            rematch_files.append((filepath, round_num))
    
    # Sort by round number
    rematch_files.sort(key=lambda x: x[1])
    
    return base_file, rematch_files


def merge_records(all_records: List[Dict]) -> List[Dict]:
    """
    Merge records, deduplicate, prefer later round results.
    
    Args:
        all_records: All records list
    
    Returns:
        Deduplicated records list, sorted by round descending
    """
    # Group by uuid
    uuid_to_records: Dict[str, List[Dict]] = {}
    for record in all_records:
        uuid = record["uuid"]
        if uuid not in uuid_to_records:
            uuid_to_records[uuid] = []
        uuid_to_records[uuid].append(record)
    
    # For each uuid, choose the record with the highest round
    merged = []
    for uuid, records in uuid_to_records.items():
        # Sort by round descending, take the first (max round)
        records.sort(key=lambda x: x["round"], reverse=True)
        merged.append(records[0])
    
    logger.info(f"Merged {len(all_records)} records into {len(merged)} unique records")
    
    # Count records per round
    round_counts = {}
    for record in merged:
        round_num = record["round"]
        round_counts[round_num] = round_counts.get(round_num, 0) + 1
    
    logger.info("Final round distribution:")
    for round_num in sorted(round_counts.keys()):
        logger.info(f"  Round {round_num}: {round_counts[round_num]} records")
    
    return merged


def main():
    args = parse_args()
    
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(input_dir, "match_results_merged.jsonl")
    
    # Find files
    base_file, rematch_files = find_files(input_dir)
    
    # Load all records
    all_records = []
    
    # Load base file (round 0)
    if os.path.exists(base_file):
        all_records.extend(load_file(base_file, 0))
    else:
        logger.warning(f"Base file not found: {base_file}")
    
    # Load all rematch files
    for filepath, round_num in rematch_files:
        all_records.extend(load_file(filepath, round_num))
    
    if not all_records:
        logger.error("No records loaded. Exiting.")
        return
    
    # Merge and deduplicate
    merged_records = merge_records(all_records)
    
    # Sort by uuid and output
    merged_records.sort(key=lambda x: x["uuid"])
    
    # Write to output file
    logger.info(f"Writing {len(merged_records)} records to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"✅ Merge completed. Output: {output_path}")


if __name__ == "__main__":
    main()
