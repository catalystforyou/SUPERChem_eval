#!/usr/bin/env python3
"""
Extract uuid and parsed fields from JSONL file with ability tags
Keep only these two fields for subsequent analysis
"""

import argparse
import json
import os
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter JSONL file to keep only uuid and parsed fields."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path (e.g., match_results_*_tagged.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output JSONL file path. If not specified, will use input filename with '_filtered' suffix.",
    )
    return parser.parse_args()


def filter_fields(input_path: str, output_path: str) -> None:
    """
    Read JSONL file, keep only uuid and parsed fields
    
    Args:
        input_path: Input file path
        output_path: Output file path
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Reading from {input_path}")
    
    total_count = 0
    filtered_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                total_count += 1
                
                # Extract uuid and parsed fields
                filtered_record = {
                    "uuid": record.get("uuid"),
                    "parsed": record.get("parsed"),
                }
                
                # Write to output file
                f_out.write(json.dumps(filtered_record, ensure_ascii=False) + "\n")
                filtered_count += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Failed to parse JSON - {e}")
                continue
    
    logger.info(f"✅ Filtered {filtered_count}/{total_count} records")
    logger.info(f"Output saved to {output_path}")


def main() -> None:
    args = parse_args()
    
    input_path = args.input
    
    # If output path not specified, auto generate
    if args.output:
        output_path = args.output
    else:
        input_file = Path(input_path)
        output_path = str(
            input_file.parent / f"{input_file.stem}_output{input_file.suffix}"
        )
    
    filter_fields(input_path, output_path)


if __name__ == "__main__":
    main()
