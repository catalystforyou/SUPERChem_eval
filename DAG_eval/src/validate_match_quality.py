#!/usr/bin/env python3
"""
Validate match_DAG result quality.
Use LLM to check if rules in match_prompt_v4.md are strictly followed.
Generate quantitative metrics and problem summary documents.
"""

import argparse
import json
import os
import time
import threading
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Validate match DAG quality.")
    parser.add_argument("--match-results", type=str, required=True, help="Path to match results JSONL file")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions file")
    parser.add_argument("--answers", type=str, required=True, help="Path to answers file")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for reports")
    parser.add_argument("--model", type=str, required=True, help="Model name for validation")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--prompt-template", type=str, default=None, help="Path to prompt template file (default: prompts/validation_prompt.md)")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to validate")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--stagger-delay", type=float, default=2.0, help="Delay in seconds between starting each worker (to avoid API rate limits)")
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


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template"""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_validation_prompt(
    prompt_template: str,
    question: str,
    options: str,
    ground_truth_analysis: str,
    ground_truth_graph: Dict[str, Any],
    model_answer: str,
    extracted_graph: Dict[str, Any],
    matches: List[Dict[str, Any]],
) -> str:
    """Build prompt for validating match quality"""
    
    prompt = prompt_template.format(
        question=question,
        options=options,
        ground_truth_analysis=ground_truth_analysis,
        ground_truth_graph=json.dumps(ground_truth_graph, ensure_ascii=False, indent=2),
        model_answer=model_answer,
        extracted_graph=json.dumps(extracted_graph, ensure_ascii=False, indent=2),
        matches=json.dumps(matches, ensure_ascii=False, indent=2),
    )
    return prompt


def call_llm_for_validation(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Call LLM to validate match quality"""
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
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "success": True,
                    "validation": result,
                    "raw_response": content,
                }
            else:
                return {
                    "success": False,
                    "error": "No JSON found in response",
                    "raw_response": content,
                }
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(exponential_backoff_with_jitter(attempt, base=2.0))
            else:
                return {
                    "success": False,
                    "error": str(e),
                }
    
    return {"success": False, "error": "Max retries reached"}


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
    raise ValueError("Invalid graph.")


def load_processed_uuids(detail_path: str) -> set:
    """Load processed UUIDs"""
    if not os.path.exists(detail_path):
        return set()
    
    processed = set()
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uuid = obj.get("uuid")
                if uuid and obj.get("validation_status") in ["completed", "skipped"]:
                    processed.add(uuid)
            except json.JSONDecodeError:
                continue
    return processed


def validate_single_record(
    match_record: Dict[str, Any],
    question_row: pd.Series,
    prompt_template: str,
    client: OpenAI,
    model: str,
    language: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Validate single match record"""
    
    uuid = match_record.get("uuid")
    
    # Basic check
    # if not match_record.get("status"):
    #     return {
    #         "uuid": uuid,
    #         "validation_status": "skipped",
    #         "reason": "extraction_failed",
    #     }
    
    parsed = match_record.get("parsed")
    if not parsed or not isinstance(parsed, dict):
        return {
            "uuid": uuid,
            "validation_status": "skipped",
            "reason": "no_parsed_data",
        }
    
    # Prepare data
    question_field = f"question_{language}"
    options_field = f"options_{language}"
    analysis_field = f"explanation_{language}"
    
    question = question_row.get(question_field, "")
    options = build_options_str(question_row.get(options_field))
    ground_truth_analysis = question_row.get(analysis_field, "")
    ground_truth_graph = normalize_graph(question_row.get("ground_truth_graph"))
    model_answer = question_row.get("llm_output", "")
    
    extracted_graph = {
        "nodes": parsed.get("nodes", []),
        "edges": parsed.get("edges", []),
    }
    matches = parsed.get("matches", [])
    
    # Build prompt
    prompt = build_validation_prompt(
        prompt_template=prompt_template,
        question=question,
        options=options,
        ground_truth_analysis=ground_truth_analysis,
        ground_truth_graph=ground_truth_graph,
        model_answer=model_answer,
        extracted_graph=extracted_graph,
        matches=matches,
    )
    
    # Call LLM to validate
    result = call_llm_for_validation(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    if result.get("success"):
        validation = result.get("validation", {})
        return {
            "uuid": uuid,
            "validation_status": "completed",
            "validation_result": validation,
            "raw_response": result.get("raw_response", ""),
        }
    else:
        return {
            "uuid": uuid,
            "validation_status": "failed",
            "error": result.get("error", ""),
        }


def validate_worker(
    match_record: Dict[str, Any],
    question_row_dict: Dict[str, Any],
    prompt_template: str,
    model_cfg: Dict[str, Any],
    model: str,
    language: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Worker function for multiprocessing"""
    # Create OpenAI client in worker process
    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg["api_key"])
    
    # Convert dict back to Series
    question_row = pd.Series(question_row_dict)
    
    return validate_single_record(
        match_record=match_record,
        question_row=question_row,
        prompt_template=prompt_template,
        client=client,
        model=model,
        language=language,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )


def generate_quantitative_report(
    validation_results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Generate quantitative metrics report"""
    
    # Collect statistics
    total = len(validation_results)
    completed = sum(1 for r in validation_results if r.get("validation_status") == "completed")
    skipped = sum(1 for r in validation_results if r.get("validation_status") == "skipped")
    failed = sum(1 for r in validation_results if r.get("validation_status") == "failed")
    
    # Quality distribution
    quality_counter = Counter()
    issue_type_counter = Counter()
    severity_counter = Counter()
    
    total_matches = 0
    total_problematic = 0
    total_missing = 0
    total_false_null = 0
    total_hallucinations = 0
    
    for result in validation_results:
        if result.get("validation_status") != "completed":
            continue
        
        validation = result.get("validation_result", {})
        quality = validation.get("overall_quality", "unknown")
        quality_counter[quality] += 1
        
        total_matches += validation.get("total_matches", 0)
        
        problematic = validation.get("problematic_matches", [])
        total_problematic += len(problematic)
        for p in problematic:
            issue_type_counter[p.get("issue_type", "unknown")] += 1
            severity_counter[p.get("severity", "unknown")] += 1
        
        total_missing += len(validation.get("missing_matches", []))
        total_false_null += len(validation.get("false_null_matches", []))
        total_hallucinations += len(validation.get("hallucination_issues", []))
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("Match DAG Quality Validation Report")
    report.append("=" * 80)
    report.append("")
    
    report.append("## Overall Statistics")
    report.append(f"Total records validated: {total}")
    report.append(f"  - Completed: {completed} ({completed/total*100:.1f}%)")
    report.append(f"  - Skipped: {skipped} ({skipped/total*100:.1f}%)")
    report.append(f"  - Failed: {failed} ({failed/total*100:.1f}%)")
    report.append("")
    
    report.append("## Quality Distribution")
    for quality, count in quality_counter.most_common():
        report.append(f"  - {quality}: {count} ({count/completed*100:.1f}%)")
    report.append("")
    
    report.append("## Match Statistics")
    report.append(f"Total matches examined: {total_matches}")
    report.append(f"Total problematic matches: {total_problematic}")
    if total_matches > 0:
        report.append(f"Problematic match rate: {total_problematic/total_matches*100:.1f}%")
    report.append(f"Total missing matches: {total_missing}")
    report.append(f"Total false null matches: {total_false_null}")
    report.append(f"Total hallucination issues: {total_hallucinations}")
    report.append("")
    
    report.append("## Issue Type Breakdown")
    for issue_type, count in issue_type_counter.most_common():
        report.append(f"  - {issue_type}: {count}")
    report.append("")
    
    report.append("## Severity Breakdown")
    for severity, count in severity_counter.most_common():
        report.append(f"  - {severity}: {count}")
    report.append("")
    
    report.append("=" * 80)
    
    # Write to file
    report_text = "\n".join(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Quantitative report saved to {output_path}")
    print("\n" + report_text)


def generate_problem_summary(
    validation_results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Generate problem summary document"""
    
    problems_by_type = defaultdict(list)
    
    for result in validation_results:
        if result.get("validation_status") != "completed":
            continue
        
        uuid = result.get("uuid")
        validation = result.get("validation_result", {})
        
        # Collect various problem types
        for problem in validation.get("problematic_matches", []):
            problems_by_type[problem.get("issue_type", "unknown")].append({
                "uuid": uuid,
                "h_id": problem.get("h_id"),
                "r_id": problem.get("r_id"),
                "severity": problem.get("severity"),
                "explanation": problem.get("explanation"),
                "should_be_null": problem.get("should_be_null"),
            })
        
        for problem in validation.get("missing_matches", []):
            problems_by_type["missing_match"].append({
                "uuid": uuid,
                "h_id": problem.get("h_id"),
                "potential_r_id": problem.get("potential_r_id"),
                "explanation": problem.get("explanation"),
            })
        
        for problem in validation.get("false_null_matches", []):
            problems_by_type["false_null"].append({
                "uuid": uuid,
                "h_id": problem.get("h_id"),
                "explanation": problem.get("explanation"),
            })
        
        for problem in validation.get("hallucination_issues", []):
            problems_by_type["hallucination"].append({
                "uuid": uuid,
                "type": problem.get("type"),
                "description": problem.get("description"),
            })
    
    # Generate Markdown document
    doc = []
    doc.append("# Match DAG Quality Issues - Detailed Summary\n")
    doc.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    doc.append("---\n")
    
    for issue_type, problems in sorted(problems_by_type.items()):
        doc.append(f"## {issue_type.replace('_', ' ').title()}\n")
        doc.append(f"**Total occurrences**: {len(problems)}\n")
        
        # Group by severity (if available)
        if issue_type not in ["missing_match", "false_null", "hallucination"]:
            severity_groups = defaultdict(list)
            for p in problems:
                severity_groups[p.get("severity", "unknown")].append(p)
            
            for severity in ["critical", "major", "minor", "unknown"]:
                if severity not in severity_groups:
                    continue
                
                doc.append(f"\n### Severity: {severity} ({len(severity_groups[severity])} cases)\n")
                
                # Show top 10 examples
                for i, problem in enumerate(severity_groups[severity][:10], 1):
                    doc.append(f"**Example {i}**\n")
                    doc.append(f"- UUID: `{problem['uuid']}`\n")
                    doc.append(f"- Match: `{problem['h_id']}` → `{problem['r_id']}`\n")
                    doc.append(f"- Should be null: {problem.get('should_be_null')}\n")
                    doc.append(f"- Explanation: {problem['explanation']}\n")
                    doc.append("\n")
        else:
            # Show top 20 examples
            for i, problem in enumerate(problems[:20], 1):
                doc.append(f"\n**Example {i}**\n")
                doc.append(f"- UUID: `{problem['uuid']}`\n")
                for key, value in problem.items():
                    if key != "uuid":
                        doc.append(f"- {key}: {value}\n")
                doc.append("\n")
        
        doc.append("\n---\n")
    
    doc_text = "".join(doc)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc_text)
    
    logger.info(f"Problem summary saved to {output_path}")


def main() -> None:
    args = parse_args()
    
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    
    try:
        config = load_config(str(config_path))
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Find model config
    model_cfg = None
    for item in config.get("model_list", []):
        if item.get("model") == args.model:
            model_cfg = item
            break
    if not model_cfg:
        logger.error(f"Model {args.model} not found in config.yaml")
        return
    
    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg["api_key"])
    
    # Load prompt template
    if args.prompt_template:
        prompt_template_path = args.prompt_template
    else:
        # Default path: ../prompts/validation_prompt.md relative to script
        prompt_template_path = str(script_dir.parent / "prompts" / "validation_prompt.md")
    
    try:
        prompt_template = load_prompt_template(prompt_template_path)
        logger.info(f"Loaded prompt template from {prompt_template_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Load data
    logger.info("Loading data...")
    match_records = []
    with open(args.match_results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                match_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    questions_df = load_data(args.questions)
    answers_df = load_data(args.answers)
    ground_truth_df = load_data(args.ground_truth)
    
    # Merge data
    merged = questions_df.merge(answers_df, on="uuid", how="inner", suffixes=("", "_ans"))
    merged = merged.merge(ground_truth_df[["uuid", "ground_truth_graph"]], on="uuid", how="inner")
    
    # Match records and questions
    uuid_to_row = {row["uuid"]: row for _, row in merged.iterrows()}
    valid_records = [r for r in match_records if r.get("uuid") in uuid_to_row]
    
    logger.info(f"Loaded {len(valid_records)} valid match records")
    
    # Sample
    import random
    if args.sample_size and len(valid_records) > args.sample_size:
        valid_records = random.sample(valid_records, args.sample_size)
        logger.info(f"Sampled {args.sample_size} records for validation")
    
    if args.limit:
        valid_records = valid_records[:args.limit]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "validation_details.jsonl")
    
    # Resume support
    if args.resume:
        processed_uuids = load_processed_uuids(detail_path)
        valid_records = [r for r in valid_records if r.get("uuid") not in processed_uuids]
        logger.info(f"Resuming: {len(valid_records)} records remaining")
    
    if len(valid_records) == 0:
        logger.info("No records to process")
        # If no records to process, still generate report
        validation_results = []
        if os.path.exists(detail_path):
            with open(detail_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            validation_results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    else:
        # Validate records
        validation_results = []
        
        # Open file for append or write
        mode = "a" if args.resume else "w"
        f_out = open(detail_path, mode, encoding="utf-8")
        
        try:
            if args.workers > 1:
                # Parallel processing
                logger.info(f"Processing with {args.workers} workers")
                logger.info(f"Staggered startup: {args.stagger_delay}s delay between workers")
                
                # Prepare task list
                tasks = []
                for match_record in valid_records:
                    uuid = match_record.get("uuid")
                    question_row = uuid_to_row[uuid]
                    question_row_dict = question_row.to_dict()
                    
                    tasks.append((
                        match_record,
                        question_row_dict,
                        prompt_template,
                        model_cfg,
                        args.model,
                        args.language,
                        args.temperature,
                        args.timeout,
                        args.max_retries,
                    ))
                
                # Use process pool, submit tasks gradually to avoid API rate limit
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {}
                    futures_lock = threading.Lock()
                    
                    # Submit tasks gradually
                    def submit_tasks_gradually():
                        for i, task in enumerate(tasks):
                            # Start first N tasks (N=workers) gradually to avoid triggering API simultaneously
                            if i < args.workers and args.stagger_delay > 0:
                                time.sleep(args.stagger_delay)
                                logger.info(f"Starting worker {i+1}/{args.workers}")
                            
                            future = executor.submit(validate_worker, *task)
                            with futures_lock:
                                futures[future] = task[0].get("uuid")
                    
                    # Submit tasks in background thread
                    submit_thread = threading.Thread(target=submit_tasks_gradually)
                    submit_thread.start()
                    
                    # Wait for tasks to complete
                    completed_count = 0
                    with tqdm(total=len(tasks), desc="Validating") as pbar:
                        while completed_count < len(tasks):
                            # Get current futures copy
                            with futures_lock:
                                current_futures = dict(futures)
                            
                            # Check completed futures
                            for future in list(current_futures.keys()):
                                if future.done():
                                    try:
                                        result = future.result()
                                        validation_results.append(result)
                                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                                        f_out.flush()
                                    except Exception as e:
                                        uuid = current_futures[future]
                                        logger.error(f"UUID {uuid} failed: {e}")
                                        error_result = {
                                            "uuid": uuid,
                                            "validation_status": "failed",
                                            "error": str(e),
                                        }
                                        validation_results.append(error_result)
                                        f_out.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                                        f_out.flush()
                                    
                                    with futures_lock:
                                        del futures[future]
                                    completed_count += 1
                                    pbar.update(1)
                            
                            time.sleep(0.1)  # Brief sleep to avoid high CPU usage
                    
                    submit_thread.join()
            else:
                # Sequential processing
                for match_record in tqdm(valid_records, desc="Validating"):
                    uuid = match_record.get("uuid")
                    question_row = uuid_to_row[uuid]
                    
                    result = validate_single_record(
                        match_record=match_record,
                        question_row=question_row,
                        prompt_template=prompt_template,
                        client=client,
                        model=args.model,
                        language=args.language,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                    )
                    validation_results.append(result)
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
        finally:
            f_out.close()
        
        logger.info(f"Detailed results saved to {detail_path}")
        
        # If resume mode, need to load all results for report generation
        if args.resume:
            all_results = []
            with open(detail_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            validation_results = all_results
    
    # Generate quantitative report
    report_path = os.path.join(args.output_dir, "quantitative_report.txt")
    generate_quantitative_report(validation_results, report_path)
    
    # Generate problem summary
    summary_path = os.path.join(args.output_dir, "problem_summary.md")
    generate_problem_summary(validation_results, summary_path)
    
    logger.info("✅ Validation completed!")


if __name__ == "__main__":
    main()
