#!/usr/bin/env python3
"""
Re-match DAG based on validation issues using LLM.
Uses rematch_prompt.md as the prompt template.
"""

import argparse
import json
import os
import re
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from validate_dag import validate_graph

def exponential_backoff_with_jitter(
    attempt: int,
    base: float = 1.0,
    jitter_ratio: float = 0.3,
    max_delay: float = 60.0,
) -> float:
    """Exponential backoff with random jitter."""
    delay = base * (2 ** attempt)
    jitter = random.uniform(-jitter_ratio, jitter_ratio) * delay
    return min(max_delay, max(0.0, delay + jitter))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-match DAG based on validation issues.")
    parser.add_argument("--tagged-output", type=str, required=True, help="Path to tagged output JSONL file (e.g., xxx_tagged_output.jsonl)")
    parser.add_argument("--validation-details", type=str, required=True, help="Path to validation details JSONL file")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions file")
    parser.add_argument("--answers", type=str, required=True, help="Path to answers file")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for corrected matches")
    parser.add_argument("--model", type=str, required=True, help="Model name for re-matching")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--answer-field", type=str, default="llm_output")
    parser.add_argument("--prompt-template", type=str, default="rematch_prompt.md", help="Path to rematch prompt template")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--stagger-delay", type=float, default=2.0, help="Delay in seconds between starting each worker (to avoid API rate limits)")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--max-validation-retries", type=int, default=3, help="Maximum number of validation retries (default: 3)")
    parser.add_argument("--only-quality-threshold", type=str, default=None, 
                        help="Only process records with validation quality below this threshold (e.g., 'fair,poor')")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    raise ValueError("Unsupported file format.")


def build_options_str(options_dict: Any) -> str:
    if not isinstance(options_dict, dict):
        return ""
    options = []
    for key in sorted(options_dict.keys()):
        value = options_dict[key]
        if value is None:
            break
        options.append(f"{key}: {value}")
    return "\n".join(options)


def normalize_graph(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError("Invalid graph format.")


def format_validation_issues(validation_result: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Format validation issues as string"""
    
    # Problematic matches
    problematic = validation_result.get("problematic_matches", [])
    problematic_str = ""
    if problematic:
        problematic_str = "\n".join([
            f"- **H{m['h_id']} → R{m['r_id']}**: {m['issue_type']} (severity: {m['severity']})\n"
            f"  Explanation: {m['explanation']}\n"
            f"  Should be null: {m.get('should_be_null', 'N/A')}"
            for m in problematic
        ])
    else:
        problematic_str = "None"
    
    # Missing matches
    missing = validation_result.get("missing_matches", [])
    missing_str = ""
    if missing:
        missing_str = "\n".join([
            f"- **H{m['h_id']}** could match **R{m.get('potential_r_id', '?')}**\n"
            f"  Explanation: {m['explanation']}"
            for m in missing
        ])
    else:
        missing_str = "None"
    
    # False null matches
    false_null = validation_result.get("false_null_matches", [])
    false_null_str = ""
    if false_null:
        false_null_str = "\n".join([
            f"- **H{m['h_id']}** was incorrectly marked as null\n"
            f"  Explanation: {m['explanation']}"
            for m in false_null
        ])
    else:
        false_null_str = "None"
    
    # Edge issues
    edge_issues = validation_result.get("edge_issues", [])
    edge_issues_str = ""
    if edge_issues:
        edge_issues_str = "\n".join([
            f"- **Edge {e['from_h_id']} → {e['to_h_id']}**: {e['issue_type']}\n"
            f"  Explanation: {e['explanation']}\n"
            f"  Suggested action: {e.get('suggested_action', 'N/A')}"
            for e in edge_issues
        ])
    else:
        edge_issues_str = "None"
    
    return problematic_str, missing_str, false_null_str, edge_issues_str


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template"""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_rematch_prompt(
    template: str,
    question: str,
    options: str,
    ground_truth_analysis: str,
    ground_truth_graph: Dict[str, Any],
    model_answer: str,
    extracted_dag: Dict[str, Any],
    original_matches: List[Dict[str, Any]],
    validation_result: Dict[str, Any],
) -> str:
    """Build re-matching prompt"""
    
    problematic_str, missing_str, false_null_str, edge_issues_str = format_validation_issues(validation_result)
    
    # Build JSON format for validation issues
    validation_issues_json = {
        "problematic_matches": validation_result.get("problematic_matches", []),
        "missing_matches": validation_result.get("missing_matches", []),
        "false_null_matches": validation_result.get("false_null_matches", []),
        "edge_issues": validation_result.get("edge_issues", []),
    }
    
    prompt = template.format(
        question=question,
        options=options,
        ground_truth_analysis=ground_truth_analysis,
        ground_truth_graph=json.dumps(ground_truth_graph, ensure_ascii=False, indent=2),
        model_answer=model_answer,
        extracted_dag=json.dumps(extracted_dag, ensure_ascii=False, indent=2),
        original_matches=json.dumps(original_matches, ensure_ascii=False, indent=2),
        validation_issues=json.dumps(validation_issues_json, ensure_ascii=False, indent=2),
        problematic_matches_detail=problematic_str,
        missing_matches_detail=missing_str,
        false_null_matches_detail=false_null_str,
        edge_issues_detail=edge_issues_str,
    )
    
    return prompt


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON response"""
    # First try to match JSON in markdown code block
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from code block: {e}")
    
    # Try to match entire JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON object: {e}")
            # Try to fix common JSON format issues
            # Remove possible markdown residue
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    raise ValueError(f"No valid JSON object found in response. Response preview: {text[:500]}")


def call_llm_for_rematch(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Call LLM for re-matching"""
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                max_tokens=32000,
                extra_body={
                    "thinking": {
                        "type": "enabled",
                        "level": "high"
                    }
                }
            )
            
            content = response.choices[0].message.content or ""
            
            # Parse JSON response
            try:
                result = parse_json_response(content)
                return {
                    "success": True,
                    "rematch_result": result,
                    "raw_response": content,
                }
            except Exception as parse_error:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - JSON parse failed: {parse_error}")
                if attempt < max_retries - 1:
                    time.sleep(exponential_backoff_with_jitter(attempt, base=2.0))
                else:
                    return {
                        "success": False,
                        "error": f"JSON parse error: {str(parse_error)}",
                        "raw_response": content,
                    }
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} - API call failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(exponential_backoff_with_jitter(attempt, base=2.0))
            else:
                return {
                    "success": False,
                    "error": f"API error: {str(e)}",
                }
    
    return {"success": False, "error": "Max retries reached"}


def merge_corrected_matches(
    original_matches: List[Dict[str, Any]],
    corrected_matches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge original and corrected matches"""
    # Create h_id to match mapping
    corrected_map = {m["h_id"]: m["r_id"] for m in corrected_matches}
    
    # Update original matches
    merged = []
    for match in original_matches:
        h_id = match["h_id"]
        if h_id in corrected_map:
            merged.append({"h_id": h_id, "r_id": corrected_map[h_id]})
        else:
            merged.append(match)
    
    return merged


def merge_corrected_edges(
    original_edges: List[Dict[str, Any]],
    corrected_edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge original and corrected edges"""
    # Convert edges to comparable tuples
    edge_to_action = {}
    for e in corrected_edges:
        edge_tuple = (e["from"], e["to"])
        edge_to_action[edge_tuple] = e.get("action", "add")
    
    # Process original edges
    merged = []
    for edge in original_edges:
        edge_tuple = (edge["from"], edge["to"])
        action = edge_to_action.get(edge_tuple)
        
        if action == "remove":
            # Remove this edge
            continue
        elif action == "reverse":
            # Reverse edge direction
            merged.append({"from": edge["to"], "to": edge["from"]})
        else:
            # Keep original edge
            merged.append(edge)
    
    # Add new edges
    for e in corrected_edges:
        if e.get("action") == "add":
            edge_tuple = (e["from"], e["to"])
            # Check if already exists
            if not any((edge["from"], edge["to"]) == edge_tuple for edge in merged):
                merged.append({"from": e["from"], "to": e["to"]})
    
    return merged


def process_single_record(
    record: Dict[str, Any],
    validation: Dict[str, Any],
    question_row: pd.Series,
    answer_row: pd.Series,
    gt_row: pd.Series,
    prompt_template: str,
    client: OpenAI,
    model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    max_validation_retries: int,
    language: str,
    answer_field: str,
) -> Dict[str, Any]:
    """Process single record re-matching"""
    
    uuid = record["uuid"]
    
    # Check if validation has issues needing correction
    validation_result = validation.get("validation_result", {})
    has_issues = (
        len(validation_result.get("problematic_matches", [])) > 0 or
        len(validation_result.get("missing_matches", [])) > 0 or
        len(validation_result.get("false_null_matches", [])) > 0 or
        len(validation_result.get("edge_issues", [])) > 0
    )
    
    if not has_issues:
        logger.debug(f"UUID {uuid}: No issues found, skipping rematch")
        # Keep original parsed data unchanged
        return {
            "uuid": uuid,
            "parsed": record["parsed"],  # Keep original parsed field
            "rematch_status": "skipped",
            "reason": "no_issues_found",
        }
    
    # Prepare data
    question = question_row[f"question_{language}"]
    options = build_options_str(question_row.get(f"options_{language}"))
    
    # Support different ground truth data formats
    if f"analysis_{language}" in gt_row:
        ground_truth_analysis = gt_row[f"analysis_{language}"]
    elif "response" in gt_row:
        ground_truth_analysis = gt_row["response"]
    else:
        ground_truth_analysis = "No analysis available"
    
    # Support different graph field names
    if "reasoning_graph" in gt_row:
        ground_truth_graph = normalize_graph(gt_row["reasoning_graph"])
    elif "ground_truth_graph" in gt_row:
        ground_truth_graph = normalize_graph(gt_row["ground_truth_graph"])
    else:
        raise ValueError(f"UUID {uuid}: No ground truth graph found in data")
    model_answer = answer_row[answer_field]
    extracted_dag = record["parsed"]
    original_matches = record["parsed"]["matches"]
    
    # Build prompt
    prompt_base = build_rematch_prompt(
        prompt_template,
        question,
        options,
        ground_truth_analysis,
        ground_truth_graph,
        model_answer,
        extracted_dag,
        original_matches,
        validation_result,
    )
    
    validation_attempt = 0
    validation_errors: List[str] = []
    while validation_attempt <= max_validation_retries:
        prompt = prompt_base
        if validation_errors:
            prompt = (
                prompt_base
                + "\n\nValidation errors from previous output:\n- "
                + "\n- ".join(validation_errors[:8])
                + "\nPlease correct the JSON strictly. Node ids must be H<integer>, and edges must reference existing nodes."
            )
        
        # Call LLM
        llm_result = call_llm_for_rematch(
            client, model, prompt, temperature, timeout, max_retries
        )
        
        if not llm_result["success"]:
            error_msg = llm_result.get("error", "Unknown error")
            logger.error(f"UUID {uuid}: LLM call failed - {error_msg}")
            # Keep original parsed data unchanged
            return {
                "uuid": uuid,
                "parsed": record["parsed"],  # Keep original parsed field
                "rematch_status": "failed",
                "error": error_msg,
                "raw_response": llm_result.get("raw_response", ""),
            }
        
        # Merge corrected matches
        try:
            rematch_result = llm_result["rematch_result"]
            
            # Check if rematch_result is a dict
            if not isinstance(rematch_result, dict):
                logger.error(f"UUID {uuid}: rematch_result is not a dict, type={type(rematch_result)}, value={rematch_result}")
                return {
                    "uuid": uuid,
                    "parsed": record["parsed"],  # Keep original parsed field
                    "rematch_status": "failed",
                    "error": f"Invalid rematch_result type: {type(rematch_result)}",
                    "raw_response": llm_result.get("raw_response", ""),
                }
            
            corrected_matches = rematch_result.get("corrected_matches", [])
            final_matches = merge_corrected_matches(original_matches, corrected_matches)
            
            # Merge corrected edges
            original_edges = extracted_dag.get("edges", [])
            corrected_edges = rematch_result.get("corrected_edges", [])
            final_edges = merge_corrected_edges(original_edges, corrected_edges)
            
            # Build complete parsed field for subsequent validation
            parsed_data = {
                "nodes": extracted_dag.get("nodes", []),
                "edges": final_edges,
                "matches": final_matches,
            }
            
            # DAG structure validation to avoid subsequent analysis errors
            validation_errors = validate_graph(parsed_data, require_matches=True)
            if not validation_errors:
                return {
                    "uuid": uuid,
                    "parsed": parsed_data,  # Add parsed field for validation
                    "rematch_status": "completed",
                    "original_matches": original_matches,
                    "corrected_matches": corrected_matches,
                    "final_matches": final_matches,
                    "original_edges": original_edges,
                    "corrected_edges": corrected_edges,
                    "final_edges": final_edges,
                    "rematch_summary": rematch_result.get("summary", {}),
                    "raw_response": llm_result.get("raw_response", ""),
                }
            
            validation_attempt += 1
            if "validation_failures" not in record:
                record["validation_failures"] = []
            record["validation_failures"].append({
                "attempt": validation_attempt,
                "errors": validation_errors[:8],
            })
            
            if validation_attempt > max_validation_retries:
                logger.error(
                    f"UUID {uuid}: Max validation retries reached. Last errors: {validation_errors[:3]}"
                )
                return {
                    "uuid": uuid,
                    "parsed": record["parsed"],  # Keep original parsed field
                    "rematch_status": "failed",
                    "error": "invalid_dag: " + "; ".join(validation_errors[:3]),
                    "raw_response": llm_result.get("raw_response", ""),
                }
        except Exception as e:
            logger.error(f"UUID {uuid}: Error processing rematch result - {e}")
            logger.error(f"UUID {uuid}: raw_response preview: {llm_result.get('raw_response', '')[:500]}")
            return {
                "uuid": uuid,
                "parsed": record["parsed"],  # Keep original parsed field
                "rematch_status": "failed",
                "error": str(e),
                "raw_response": llm_result.get("raw_response", ""),
            }


def load_processed_uuids(output_path: str) -> set:
    """Load processed UUIDs"""
    if not os.path.exists(output_path):
        return set()
    processed = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            processed.add(data["uuid"])
    return processed


def worker_function(args):
    """Worker function for multi-threading"""
    (
        record,
        validation,
        question_row,
        answer_row,
        gt_row,
        prompt_template,
        client,
        model,
        temperature,
        timeout,
        max_retries,
        max_validation_retries,
        language,
        answer_field,
    ) = args
    
    try:
        return process_single_record(
            record,
            validation,
            question_row,
            answer_row,
            gt_row,
            prompt_template,
            client,
            model,
            temperature,
            timeout,
            max_retries,
            max_validation_retries,
            language,
            answer_field,
        )
    except Exception as e:
        uuid = record.get("uuid", "unknown")
        logger.error(f"Worker exception for UUID {uuid}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "uuid": uuid,
            "parsed": record.get("parsed", {}),  # Keep original parsed field
            "rematch_status": "failed",
            "error": f"Worker exception: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def main():
    args = parse_args()
    # logger.remove()
    # logger.add(lambda msg: print(msg, end=""), level=args.log_level)
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    
    # Initialize LLM client
    model_config = next((m for m in config["model_list"] if m["model"] == args.model), None)
    if not model_config:
        raise ValueError(f"Model {args.model} not found in config.")
    
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
    )
    
    # Load data
    logger.info("Loading data...")
    tagged_data = load_data(args.tagged_output)
    validation_data = load_data(args.validation_details)
    questions_df = load_data(args.questions)
    answers_df = load_data(args.answers)
    gt_df = load_data(args.ground_truth)
    
    # Create UUID index
    questions_dict = questions_df.set_index("uuid").to_dict("index")
    answers_dict = answers_df.set_index("uuid").to_dict("index")
    gt_dict = gt_df.set_index("uuid").to_dict("index")
    validation_dict = validation_data.set_index("uuid").to_dict("index")
    
    # Load prompt template
    logger.info(f"Loading prompt template from {args.prompt_template}")
    prompt_template = load_prompt_template(args.prompt_template)
    
    # Prepare output
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create when output path contains directory
        os.makedirs(output_dir, exist_ok=True)
    processed_uuids = load_processed_uuids(args.output) if args.resume else set()
    
    # Filter records to process
    records_to_process = []
    for _, row in tagged_data.iterrows():
        uuid = row["uuid"]
        if uuid in processed_uuids:
            continue
        if uuid not in validation_dict:
            logger.warning(f"UUID {uuid} not found in validation data, skipping")
            continue
        
        validation = validation_dict[uuid]
        
        # Filter by quality threshold
        if args.only_quality_threshold:
            quality = validation.get("validation_result", {}).get("overall_quality", "")
            threshold_list = [q.strip() for q in args.only_quality_threshold.split(",")]
            if quality not in threshold_list:
                continue
        
        records_to_process.append({
            "record": row.to_dict(),
            "validation": validation,
            "question_row": questions_dict[uuid],
            "answer_row": answers_dict[uuid],
            "gt_row": gt_dict[uuid],
        })
    
    if args.limit:
        records_to_process = records_to_process[:args.limit]
    
    logger.info(f"Processing {len(records_to_process)} records...")
    
    # Prepare worker parameters
    worker_args = [
        (
            item["record"],
            item["validation"],
            item["question_row"],
            item["answer_row"],
            item["gt_row"],
            prompt_template,
            client,
            args.model,
            args.temperature,
            args.timeout,
            args.max_retries,
            args.max_validation_retries,
            args.language,
            args.answer_field,
        )
        for item in records_to_process
    ]
    
    # Process records
    output_file = open(args.output, "a", encoding="utf-8")
    
    try:
        if args.workers == 1:
            # Single thread processing
            for worker_arg in tqdm(worker_args, desc="Re-matching"):
                result = worker_function(worker_arg)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
        else:
            # Multi-thread processing, use gradual startup to avoid API rate limit
            logger.info(f"Processing with {args.workers} workers")
            logger.info(f"Staggered startup: {args.stagger_delay}s delay between workers")
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                futures_lock = threading.Lock()
                
                # Submit tasks gradually
                def submit_tasks_gradually():
                    for i, arg in enumerate(worker_args):
                        # Start first N tasks (N=workers) gradually to avoid triggering API simultaneously
                        if i < args.workers and args.stagger_delay > 0:
                            time.sleep(args.stagger_delay)
                            logger.info(f"Starting worker {i+1}/{args.workers}")
                        
                        future = executor.submit(worker_function, arg)
                        with futures_lock:
                            # Use uuid of first parameter (record dict) as identifier
                            futures[future] = arg[0].get("uuid", f"task_{i}")
                
                # Submit tasks in background thread
                submit_thread = threading.Thread(target=submit_tasks_gradually)
                submit_thread.start()
                
                # Wait for tasks to complete
                completed_count = 0
                with tqdm(total=len(worker_args), desc="Re-matching") as pbar:
                    while completed_count < len(worker_args):
                        # Get current futures copy
                        with futures_lock:
                            current_futures = dict(futures)
                        
                        # Check completed futures
                        for future in list(current_futures.keys()):
                            if future.done():
                                try:
                                    result = future.result()
                                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                                    output_file.flush()
                                except Exception as e:
                                    uuid = current_futures[future]
                                    logger.error(f"UUID {uuid} failed: {e}")
                                    error_result = {
                                        "uuid": uuid,
                                        "rematch_status": "failed",
                                        "error": str(e),
                                    }
                                    output_file.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                                    output_file.flush()
                                
                                with futures_lock:
                                    del futures[future]
                                completed_count += 1
                                pbar.update(1)
                        
                        time.sleep(0.1)  # Brief sleep to avoid high CPU usage
                
                submit_thread.join()
    finally:
        output_file.close()
    
    logger.info(f"Re-matching completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
