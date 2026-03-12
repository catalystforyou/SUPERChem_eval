import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from loguru import logger
import os
import re
import time
import random
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yaml
from openai import OpenAI
from tqdm import tqdm

from validate_dag import validate_graph


DEFAULT_CONFIG = {
    "model_list": [
        {
            "model": "your-model-name",
            "base_url": "your-api-base-url",
            "api_key": "your-api-key",
        }
    ],
    "mol_compare": {
        "url": "https://pkumdl.top/chemdraw/api/batch_mol_compare",
        "api_key": "your-api-key",
        "timeout": 15,
    },
}

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
    parser = argparse.ArgumentParser(description="Extract and match model DAG.")
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--answer-field", type=str, default="llm_output")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning-effort", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1, no parallelism)")
    parser.add_argument("--stagger-delay", type=float, default=2.0, help="Delay in seconds between starting each worker (to avoid API rate limits)")
    parser.add_argument(
        "--no-validate-and-retry",
        action="store_false",
        dest="validate_and_retry",
        help="Disable validation and retry for incomplete or invalid DAG results",
    )
    parser.add_argument("--max-validation-retries", type=int, default=3, help="Maximum number of validation retries (default: 3)")
    parser.set_defaults(validate_and_retry=True)
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, allow_unicode=False)
        raise FileNotFoundError(f"Config not found. A template was created at {config_path}.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
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


def parse_json_response(text: str) -> Dict[str, Any]:
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in response.")
        json_str = match.group(0)
    return json.loads(json_str)


def normalize_graph(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError("Invalid ground_truth_graph.")


def clean_graph_for_prompt(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean graph by removing match_score and match_strategy from nodes,
    but keeping knowledge_tags and ability_tags.
    """
    cleaned_graph = {
        "nodes": [],
        "edges": graph.get("edges", [])
    }
    
    for node in graph.get("nodes", []):
        cleaned_node = {}
        for key, value in node.items():
            # Exclude match_score and match_strategy
            if key not in ["match_score", "match_strategy", "matched_checkpoint_idx"]:
                cleaned_node[key] = value
        cleaned_graph["nodes"].append(cleaned_node)
    
    return cleaned_graph


def batch_compare_molecules(url: str, api_key: str, pairs: List[Dict[str, str]], timeout: int) -> Dict[str, Any]:
    """
    Batch compare molecules using the new API.
    
    Args:
        url: API endpoint URL
        api_key: API key for authentication
        pairs: List of dicts with keys: mol1, mol2, pair_id
        timeout: Request timeout in seconds
    
    Returns:
        API response with results for all pairs
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"pairs": pairs}
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def call_llm_with_tools(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    timeout: int,
    reasoning_effort: Optional[str],
    tool_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "batch_compare_molecules",
                "description": "Batch compare multiple pairs of chemical entities to verify if they represent the same molecule. Supports: SMILES vs SMILES, IUPAC name vs IUPAC name, and SMILES vs IUPAC name (cross-format comparison). Use this when you need to verify whether chemical structures in the model answer match those in the ground truth.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "description": "List of molecule pairs to compare. Each pair can use any combination of SMILES or IUPAC names.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "mol1": {
                                        "type": "string",
                                        "description": "First molecule (can be SMILES notation or IUPAC name)"
                                    },
                                    "mol2": {
                                        "type": "string",
                                        "description": "Second molecule (can be SMILES notation or IUPAC name, does not need to match mol1's format)"
                                    },
                                    "pair_id": {
                                        "type": "string",
                                        "description": "Unique identifier for this comparison pair (e.g., 'H1_vs_R2')"
                                    }
                                },
                                "required": ["mol1", "mol2", "pair_id"]
                            }
                        }
                    },
                    "required": ["pairs"],
                },
            },
        }
    ]

    tool_calls_log: List[Dict[str, Any]] = []
    reasoning_content = None  # Used to save reasoning content
    
    # First call: LLM decides whether to use tools
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        timeout=timeout,
        extra_body={
            "thinking": {
                "type": "enabled",
                "level": "high"
            }
        }
    )
    
    msg = response.choices[0].message
    
    # If LLM decides to use tools, execute tools and make second call for final answer
    if getattr(msg, "tool_calls", None):
        # Execute all tool calls and collect results
        tool_results_text = []
        for call in msg.tool_calls:
            if call.function.name != "batch_compare_molecules":
                continue
            try:
                args = json.loads(call.function.arguments)
                
                # Record request details
                request_info = {
                    "tool_call_id": call.id,
                    "function_name": call.function.name,
                    "arguments": args,
                    "api_endpoint": tool_cfg["url"],
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Execute tool calls
                start_time = time.time()
                result = batch_compare_molecules(
                    tool_cfg["url"],
                    tool_cfg["api_key"],
                    args["pairs"],
                    tool_cfg.get("timeout", 15),
                )
                elapsed_time = time.time() - start_time
                
                # Record complete call information
                tool_calls_log.append(
                    {
                        "tool_call_id": call.id,
                        "function_name": call.function.name,
                        "request": {
                            "arguments": args,
                            "api_endpoint": tool_cfg["url"],
                            "num_pairs": len(args.get("pairs", [])),
                            "timestamp": request_info["timestamp"],
                        },
                        "response": {
                            "result": result,
                            "elapsed_time_seconds": round(elapsed_time, 3),
                            "success": True,
                        },
                        "summary": {
                            "total_pairs": result.get("total", 0),
                            "successful_comparisons": result.get("success", 0),
                            "failed_comparisons": result.get("failed", 0),
                        }
                    }
                )
                
                # Format tool results as text
                tool_results_text.append(f"Tool: {call.function.name}")
                tool_results_text.append(f"Result: {json.dumps(result['results'], ensure_ascii=False, indent=2)}")
            except Exception as e:
                logger.warning(f"Tool call failed: {e}")
                
                # Record failed call information
                tool_calls_log.append(
                    {
                        "tool_call_id": call.id,
                        "function_name": call.function.name,
                        "request": {
                            "arguments": json.loads(call.function.arguments) if hasattr(call.function, "arguments") else {},
                            "api_endpoint": tool_cfg["url"],
                            "timestamp": datetime.now().isoformat(),
                        },
                        "response": {
                            "result": None,
                            "error": str(e),
                            "success": False,
                        },
                    }
                )
                
                tool_results_text.append(f"Tool: {call.function.name}")
                tool_results_text.append(f"Error: {str(e)}")
        
        # Second call: add tool results as text to new user message
        # Avoid using standard tool message format as Gemini checks thought_signature
        if tool_results_text:
            # print(f"Tool Results:\n{chr(10).join(tool_results_text)}")
            # Create new message list containing original messages and tool results
            messages_with_results = messages.copy()
            messages_with_results.append({
                "role": "user",
                "content": f"Based on the tool call results below, please continue to provide your analysis in the required JSON format.\n\nTool Results:\n{chr(10).join(tool_results_text)}"
            })
            
            # Second call: get final answer, do not pass tools and thinking parameters
            response = client.chat.completions.create(
                model=model,
                messages=messages_with_results,
                temperature=temperature,
                timeout=timeout,
                extra_body={
                    "thinking": {
                        "type": "enabled",
                        "level": "high"
                    }
                }
            )
            msg = response.choices[0].message
            
            # For tool calls, save second call reasoning content
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning_content = msg.reasoning_content
    else:
        # If no tool calls, save first call reasoning content
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
    
    # Return final response
    return {
        "content": msg.content or "",
        "tool_calls": tool_calls_log,
        "reasoning_content": reasoning_content,
        "usage": getattr(response, "usage", None),
    }


def validate_result(
    record: Dict[str, Any],
    row: pd.Series,
    ground_truth_graph: Dict[str, Any],
    language: str,
) -> tuple[bool, str]:
    """
    Validate if extraction result is complete and reasonable.
    
    Returns: (is_valid, reason)
    """
    # Check basic status
    if not record.get("status"):
        return False, "extraction_failed"
    
    parsed = record.get("parsed")
    if not parsed or not isinstance(parsed, dict):
        return False, "no_parsed_data"
    
    # Check graph structure completeness
    nodes = parsed.get("nodes", [])
    edges = parsed.get("edges", [])
    matches = parsed.get("matches", [])
    
    if not nodes or not isinstance(nodes, list):
        return False, "missing_nodes"
    
    if not isinstance(edges, list):
        return False, "missing_edges"
    
    if not matches or not isinstance(matches, list):
        return False, "missing_matches"
    
    # Check if nodes have basic fields
    for node in nodes:
        if not isinstance(node, dict):
            return False, "invalid_node_format"
        if "id" not in node or "content" not in node:
            return False, "incomplete_node_fields"
    
    # Check if matches have basic fields
    for match in matches:
        if not isinstance(match, dict):
            return False, "invalid_match_format"
        if "r_id" not in match or "h_id" not in match:
            return False, "incomplete_match_fields"
    
    # DAG structure validation to avoid subsequent analysis errors
    dag_errors = validate_graph(parsed, require_matches=True)
    if dag_errors:
        return False, "invalid_dag: " + "; ".join(dag_errors[:3])
    
    # Check suspicious cases: answer inconsistent but all ground truth nodes matched
    # Get answer fields
    answer_field_key = f"answer_{language}"
    ground_truth_answer = row.get(answer_field_key)
    llm_answer = row.get("llm_answer")
    
    if ground_truth_answer and llm_answer and ground_truth_answer != llm_answer:
        # Answer inconsistent, check if all ground truth nodes matched
        gt_nodes = ground_truth_graph.get("nodes", [])
        gt_node_ids = {n.get("id") for n in gt_nodes if n.get("id")}
        matched_gt_ids = {m.get("r_id") for m in matches if m.get("h_id")}
        
        # If all ground truth nodes matched, this may be suspicious
        if gt_node_ids and matched_gt_ids >= gt_node_ids:
            print(gt_node_ids)
            print('=====')
            print(matched_gt_ids)
            return False, "answer_mismatch_but_all_nodes_matched"
    
    return True, "valid"


def load_processed_uuids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    processed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uuid = obj.get("uuid")
                if uuid:
                    processed.add(uuid)
            except json.JSONDecodeError:
                continue
    return processed


def process_single_row(
    row: pd.Series,
    client: OpenAI,
    model: str,
    prompt_template: str,
    system_hint: str,
    question_field: str,
    options_field: str,
    analysis_field: str,
    answer_field: str,
    language: str,
    temperature: float,
    timeout: int,
    reasoning_effort: Optional[str],
    tool_cfg: Dict[str, Any],
    max_retries: int,
    validate_and_retry: bool = False,
    max_validation_retries: int = 3,
) -> Dict[str, Any]:
    """
    Process a single row (question-answer pair) and return the result.
    
    This function is designed to be called in parallel.
    
    Args:
        validate_and_retry: If True, validate results and retry if validation fails
        max_validation_retries: Maximum number of validation retries
    """
    uuid = row.get("uuid")
    record = {
        "uuid": uuid,
        "model": model,
        "language": language,
        "status": False,
        "error": None,
        "response": None,
        "parsed": None,
        "tool_calls": None,
        "reasoning_content": None,
    }

    try:
        ground_truth_graph = normalize_graph(row.get("ground_truth_graph"))
        # Clean graph to remove match_score and match_strategy before sending to LLM
        cleaned_graph = clean_graph_for_prompt(ground_truth_graph)
        
        prompt = prompt_template
        prompt = prompt.replace("{question}", row.get(question_field, ""))
        prompt = prompt.replace("{options}", build_options_str(row.get(options_field)))
        prompt = prompt.replace("{ground_truth_analysis}", row.get(analysis_field, ""))
        prompt = prompt.replace("{ground_truth_graph}", json.dumps(cleaned_graph, ensure_ascii=False))
        prompt = prompt.replace("{model_answer}", row.get(answer_field, ""))

        base_messages = [{"role": "system", "content": system_hint}, {"role": "user", "content": prompt}]
        messages = base_messages

        # Validation retry loop
        validation_attempt = 0
        while validation_attempt <= max_validation_retries:
            # Basic extraction retry
            for attempt in range(max_retries):
                try:
                    result = call_llm_with_tools(
                        client=client,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=timeout,
                        reasoning_effort=reasoning_effort,
                        tool_cfg=tool_cfg,
                    )
                    record["response"] = result["content"]
                    record["tool_calls"] = result["tool_calls"]
                    record["reasoning_content"] = result.get("reasoning_content")
                    record["usage"] = (
                        result["usage"].model_dump() if result.get("usage") is not None else None
                    )
                    record["parsed"] = parse_json_response(result["content"])
                    record["status"] = True
                    break
                except Exception as e:
                    record["error"] = str(e)
                    if attempt < max_retries - 1:
                        time.sleep(exponential_backoff_with_jitter(attempt, base=1.5))
            
            # If validation not needed or extraction failed, return directly
            if not validate_and_retry or not record.get("status"):
                break
            
            # Validate results
            is_valid, reason = validate_result(record, row, ground_truth_graph, language)
            
            if is_valid:
                # Validation passed
                break
            else:
                # Validation failed
                validation_attempt += 1
                if validation_attempt <= max_validation_retries:
                    logger.warning(
                        f"Validation failed for uuid {uuid} (attempt {validation_attempt}/{max_validation_retries}): {reason}. Retrying..."
                    )
                    # Add validation failure info to record
                    if "validation_failures" not in record:
                        record["validation_failures"] = []
                    record["validation_failures"].append({
                        "attempt": validation_attempt,
                        "reason": reason,
                    })
                    # Reset state, prepare for re-extraction
                    record["status"] = False
                    record["error"] = None
                    messages = base_messages + [{
                        "role": "user",
                        "content": (
                            "Your previous JSON failed validation.\n"
                            f"Reason: {reason}\n"
                            "Please re-output a corrected JSON that strictly follows the schema. "
                            "Node ids must be H<integer> and all edges must reference existing nodes."
                        )
                    }]
                    time.sleep(
                        exponential_backoff_with_jitter(
                            validation_attempt - 1,
                            base=2.0,
                        )
                    )
                else:
                    # Max retry count reached
                    logger.error(
                        f"Max validation retries reached for uuid {uuid}. Last failure reason: {reason}"
                    )
                    record["validation_failed"] = True
                    record["validation_failure_reason"] = reason
                    break
                    
    except Exception as e:
        record["error"] = str(e)
        logger.error(f"Error processing uuid {uuid}: {e}")

    return record


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    model_cfg = None
    for item in config.get("model_list", []):
        if item.get("model") == args.model:
            model_cfg = item
            break
    if not model_cfg:
        logger.error("Model not found in config.yaml.")
        return

    tool_cfg = config.get("mol_compare", {})
    if not tool_cfg.get("url") or not tool_cfg.get("api_key"):
        logger.error("mol_compare config is missing url or api_key.")
        return

    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg["api_key"])
    prompt_template = open(args.prompt, "r", encoding="utf-8").read()

    questions_df = load_data(args.questions)
    answers_df = load_data(args.answers)
    if "status" in answers_df.columns:
        answers_df = answers_df[answers_df["status"] == True]
    ground_truth_df = load_data(args.ground_truth)

    if "uuid" not in questions_df.columns or "uuid" not in answers_df.columns or "uuid" not in ground_truth_df.columns:
        logger.error("Questions, answers, and ground truth must contain uuid.")
        return

    merged = questions_df.merge(answers_df, on="uuid", how="inner", suffixes=("", "_ans"))
    merged = merged.merge(ground_truth_df[["uuid", "ground_truth_graph"]], on="uuid", how="inner")

    if args.resume:
        processed = load_processed_uuids(args.output)
        merged = merged[~merged["uuid"].isin(processed)]
    if args.limit:
        merged = merged.head(args.limit)

    question_field = f"question_{args.language}"
    options_field = f"options_{args.language}"
    analysis_field = f"explanation_{args.language}"

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    system_hint = (
        "You can call the tool batch_compare_molecules when you need to verify whether "
        "chemical entities in the model answer match those in the ground truth. "
        "The tool supports flexible comparisons: SMILES vs SMILES, IUPAC name vs IUPAC name, "
        "and SMILES vs IUPAC name (cross-format comparison). "
        "You can compare multiple pairs in a single call. Each pair should have a unique pair_id (e.g., 'H1_vs_R2'). "
        "IMPORTANT: When calling the tool, prefer using SMILES notation for molecules when available in the text. "
        "Avoid using non-standard representations like 'Ph-(CH2)3-O-(CH2)3-Ph'. "
        "If no molecular structures need to be compared, you don't need to call the tool."
    )

    # Prepare rows as list for parallel processing
    rows_to_process = [row for _, row in merged.iterrows()]
    
    if args.workers <= 1:
        # Sequential processing (original behavior)
        logger.info("Running in sequential mode (workers=1)")
        if args.validate_and_retry:
            logger.info(f"Validation and retry enabled (max retries: {args.max_validation_retries})")
        all_records = []
        with open(args.output, "a", encoding="utf-8") as f_out:
            for row in tqdm(rows_to_process, desc="Processing"):
                record = process_single_row(
                    row=row,
                    client=client,
                    model=args.model,
                    prompt_template=prompt_template,
                    system_hint=system_hint,
                    question_field=question_field,
                    options_field=options_field,
                    analysis_field=analysis_field,
                    answer_field=args.answer_field,
                    language=args.language,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    reasoning_effort=args.reasoning_effort,
                    tool_cfg=tool_cfg,
                    max_retries=args.max_retries,
                    validate_and_retry=args.validate_and_retry,
                    max_validation_retries=args.max_validation_retries,
                )
                all_records.append(record)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
        
        # Output statistics
        if args.validate_and_retry:
            logger.info("\n=== Validation Statistics ===")
            total_validated = sum(1 for r in all_records if "validation_failures" in r or r.get("validation_failed"))
            total_failed_validation = sum(1 for r in all_records if r.get("validation_failed"))
            logger.info(f"Total records with validation attempts: {total_validated}")
            logger.info(f"Total records failed final validation: {total_failed_validation}")
            
            # Count various validation failure reasons
            failure_reasons = {}
            for record in all_records:
                if "validation_failures" in record:
                    for failure in record["validation_failures"]:
                        reason = failure.get("reason", "unknown")
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            if failure_reasons:
                logger.info("Validation failure reasons:")
                for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  - {reason}: {count}")
    else:
        # Parallel processing
        logger.info(f"Running in parallel mode with {args.workers} workers")
        if args.validate_and_retry:
            logger.info(f"Validation and retry enabled (max retries: {args.max_validation_retries})")
        if args.stagger_delay > 0:
            logger.info(f"Staggered startup: {args.stagger_delay}s delay between workers")
        
        # Use a lock for thread-safe file writing
        import threading
        write_lock = threading.Lock()
        
        with open(args.output, "a", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks with staggered startup
                future_to_row = {}
                for i, row in enumerate(rows_to_process):
                    if i < args.workers and args.stagger_delay > 0:
                        time.sleep(args.stagger_delay)
                        logger.info(f"Starting worker {i + 1}/{args.workers}")
                    future = executor.submit(
                        process_single_row,
                        row=row,
                        client=client,
                        model=args.model,
                        prompt_template=prompt_template,
                        system_hint=system_hint,
                        question_field=question_field,
                        options_field=options_field,
                        analysis_field=analysis_field,
                        answer_field=args.answer_field,
                        language=args.language,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        reasoning_effort=args.reasoning_effort,
                        tool_cfg=tool_cfg,
                        max_retries=args.max_retries,
                        validate_and_retry=args.validate_and_retry,
                        max_validation_retries=args.max_validation_retries,
                    )
                    future_to_row[future] = row
                
                # Process completed tasks with progress bar
                all_records = []
                with tqdm(total=len(future_to_row), desc="Processing") as pbar:
                    for future in as_completed(future_to_row):
                        try:
                            record = future.result()
                            all_records.append(record)
                            # Thread-safe write
                            with write_lock:
                                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                f_out.flush()
                        except Exception as e:
                            row = future_to_row[future]
                            logger.error(f"Task failed for uuid {row.get('uuid')}: {e}")
                        finally:
                            pbar.update(1)
    
    # Output statistics
    if args.validate_and_retry and args.workers > 1:
        logger.info("\n=== Validation Statistics ===")
        total_validated = sum(1 for r in all_records if "validation_failures" in r or r.get("validation_failed"))
        total_failed_validation = sum(1 for r in all_records if r.get("validation_failed"))
        logger.info(f"Total records with validation attempts: {total_validated}")
        logger.info(f"Total records failed final validation: {total_failed_validation}")
        
        # Count various validation failure reasons
        failure_reasons = {}
        for record in all_records:
            if "validation_failures" in record:
                for failure in record["validation_failures"]:
                    reason = failure.get("reason", "unknown")
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        if failure_reasons:
            logger.info("Validation failure reasons:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
