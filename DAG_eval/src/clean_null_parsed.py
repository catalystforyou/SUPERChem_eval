#!/usr/bin/env python3
"""
Check records with null parsed field in match_results.jsonl
Move them to backup file, then regenerate match_results.jsonl without these records
This allows match_dag.py to use --resume to reprocess failed records
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean null parsed records from match results.")
    parser.add_argument("--input", type=str, required=True, help="Input match results JSONL file")
    parser.add_argument("--backup-dir", type=str, default=None, help="Directory to save backup files (default: same as input)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return
    
    # Determine backup directory
    if args.backup_dir:
        backup_dir = Path(args.backup_dir)
    else:
        backup_dir = input_path.parent
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup filename (with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{input_path.stem}_null_parsed_{timestamp}.jsonl"
    
    # Read all records
    print(f"Reading from {input_path}")
    all_records = []
    null_parsed_records = []
    error_uuids = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                all_records.append(record)
                
                # Check parsed field
                parsed = record.get("parsed")
                uuid = record.get("uuid", "unknown")
                if parsed is None:
                    null_parsed_records.append(record)
                    error_uuids.append(uuid)
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {line_num}: Failed to parse JSON - {e}", file=sys.stderr)
                continue
    
    total_count = len(all_records)
    null_count = len(null_parsed_records)
    valid_count = total_count - null_count
    
    print(f"Total records: {total_count}")
    if total_count > 0:
        print(f"Records with null parsed: {null_count} ({null_count/total_count*100:.2f}%)")
        print(f"Valid records: {valid_count} ({valid_count/total_count*100:.2f}%)")
    else:
        print("Records with null parsed: 0")
        print("Valid records: 0")
    
    if null_count == 0:
        print("✅ No null parsed records found. Nothing to clean.")
        return
    
    # Save null parsed records to backup file
    print(f"Saving {null_count} null parsed records to {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        for record in null_parsed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Keep only valid records, rewrite to original file
    print(f"Removing null parsed records from {input_path}")
    valid_records = [r for r in all_records if r.get("parsed") is not None]
    
    with open(input_path, "w", encoding="utf-8") as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"✅ Cleaned {null_count} null parsed records")
    print(f"📄 Backup saved to: {backup_path}")
    print(f"📄 Original file updated: {input_path} ({valid_count} records remaining)")
    if null_count > 0:
        failed_sample = ', '.join(error_uuids[:10])
        if null_count > 10:
            failed_sample += "..."
        print(f"⚠️  Failed UUIDs ({null_count}): {failed_sample}")
        print(f"💡 You can now run match_dag.py with --resume to process the {null_count} failed records")


if __name__ == "__main__":
    main()
