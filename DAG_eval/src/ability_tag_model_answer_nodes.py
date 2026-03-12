#!/usr/bin/env python3
"""
Add ability tags to DAG nodes extracted from model answers
Use LLM to identify ability tags for all nodes of a question at once
Use batch tagging to improve context understanding and tagging consistency
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


DEFAULT_CONFIG = {
    "model_list": [
        {
            "model": "your-model-name",
            "base_url": "your-api-base-url",
            "api_key": "your-api-key",
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag model answer DAG nodes with ability tags.")
    parser.add_argument("--match-results", type=str, required=True, help="Path to match results JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output path for tagged results")
    parser.add_argument("--model", type=str, required=True, help="Model name for tagging")
    parser.add_argument("--prompt-template", type=str, default="ability_tagging_prompt.md", help="Path to prompt template file")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds (increased for batch processing)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1 for sequential)")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt_template(path: str) -> str:
    """Load prompt template"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Extract main part of template, remove placeholders from examples
        return content


def build_tagging_prompt(
    question: str,
    options: str,
    model_answer: str,
    dag_nodes: List[Dict[str, Any]],
    dag_edges: List[Dict[str, Any]],
    prompt_template: str,
) -> str:
    """Build prompt for tagging ability tags"""
    
    # Build JSON representation of DAG
    extracted_dag = json.dumps(
        {"nodes": dag_nodes, "edges": dag_edges},
        ensure_ascii=False,
        indent=2
    )
    
    # Fill placeholders in template
    prompt = prompt_template.replace("{question}", question)
    prompt = prompt.replace("{options}", options if options else "N/A")
    prompt = prompt.replace("{model_answer}", model_answer)
    prompt = prompt.replace("{extracted_dag}", extracted_dag)
    
    return prompt


def call_llm_for_tagging(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Call LLM to add ability tags for all nodes"""
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            
            content = response.choices[0].message.content or ""
            
            # Parse JSON response - match complete JSON object
            import re
            # Try to find outermost JSON object
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
                # Verify returned result contains necessary fields
                if "node_tags" in result and isinstance(result["node_tags"], list):
                    return {
                        "success": True,
                        "node_tags": result.get("node_tags", []),
                        "overall_analysis": result.get("overall_analysis", ""),
                        "raw_response": content,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Response missing 'node_tags' field or invalid format",
                        "raw_response": content,
                    }
            else:
                return {
                    "success": False,
                    "error": "No JSON found in response",
                    "raw_response": content,
                }
                
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} - JSON decode error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return {
                    "success": False,
                    "error": f"JSON decode error: {str(e)}",
                }
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return {
                    "success": False,
                    "error": str(e),
                }
    
    return {"success": False, "error": "Max retries reached"}


def process_record(
    record: Dict[str, Any],
    client: OpenAI,
    model: str,
    prompt_template: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Process single record, add ability tags for all nodes at once"""
    
    # Check if already processed
    if record.get("nodes_tagged"):
        return record
    
    parsed = record.get("parsed")
    if not parsed or not isinstance(parsed, dict):
        logger.warning(f"UUID {record.get('uuid')}: No valid parsed data")
        return record
    
    nodes = parsed.get("nodes", [])
    edges = parsed.get("edges", [])
    if not nodes:
        logger.warning(f"UUID {record.get('uuid')}: No nodes found")
        return record
    
    # Get question info
    question = record.get("question", "")
    options = record.get("options", "")
    model_answer = record.get("model_answer", "")
    
    # Build complete prompt
    prompt = build_tagging_prompt(
        question=question,
        options=options,
        model_answer=model_answer,
        dag_nodes=nodes,
        dag_edges=edges,
        prompt_template=prompt_template,
    )
    
    # Call LLM once to tag all nodes
    result = call_llm_for_tagging(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    if not result.get("success"):
        logger.error(f"UUID {record.get('uuid')}: Tagging failed - {result.get('error')}")
        record["tagging_error"] = result.get("error", "Unknown error")
        record["tagging_raw_response"] = result.get("raw_response", "")
        return record
    
    # Parse returned tagging results
    node_tags = result.get("node_tags", [])
    overall_analysis = result.get("overall_analysis", "")
    
    # Create node_id to tags mapping
    node_id_to_tags = {}
    tagging_details = []
    for tag_item in node_tags:
        node_id = tag_item.get("node_id", "")
        ability_tags = tag_item.get("ability_tags", [])
        reasoning = tag_item.get("reasoning", "")
        
        node_id_to_tags[node_id] = {
            "ability_tags": ability_tags,
            "reasoning": reasoning,
        }
        tagging_details.append({
            "node_id": node_id,
            "ability_tags": ability_tags,
            "reasoning": reasoning,
        })
    
    # Add ability_tags for each node
    tagged_nodes = []
    for node in nodes:
        node_id = node.get("id", "")
        tagged_node = node.copy()
        
        if node_id in node_id_to_tags:
            tagged_node["ability_tags"] = node_id_to_tags[node_id]["ability_tags"]
        else:
            # If LLM did not return tag for node, record warning
            logger.warning(f"UUID {record.get('uuid')}, Node {node_id}: No tags returned by LLM")
            tagged_node["ability_tags"] = []
        
        tagged_nodes.append(tagged_node)
    
    # Update record
    record["parsed"]["nodes"] = tagged_nodes
    record["nodes_tagged"] = True
    record["tagging_details"] = tagging_details
    record["overall_analysis"] = overall_analysis
    
    return record


def load_processed_uuids(path: str) -> set:
    """Load processed UUIDs"""
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
                if obj.get("nodes_tagged"):
                    processed.add(obj.get("uuid"))
            except json.JSONDecodeError:
                continue
    return processed


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
    prompt_template_path = args.prompt_template
    prompt_template = load_prompt_template(str(prompt_template_path))
    logger.info(f"Loaded prompt template from {prompt_template_path}")
    
    # Load match results
    logger.info(f"Loading match results from {args.match_results}")
    records = []
    with open(args.match_results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
    
    logger.info(f"Loaded {len(records)} records")
    
    # Resume support
    if args.resume:
        processed = load_processed_uuids(args.output)
        records = [r for r in records if r.get("uuid") not in processed]
        logger.info(f"Resuming: {len(records)} records remaining")
    
    if args.limit:
        records = records[:args.limit]
        logger.info(f"Limited to {len(records)} records")
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process records
    mode = "a" if args.resume else "w"
    
    if args.workers <= 1:
        # Serial processing (original way)
        logger.info("Processing sequentially (workers=1)")
        with open(args.output, mode, encoding="utf-8") as f_out:
            for record in tqdm(records, desc="Tagging nodes"):
                tagged_record = process_record(
                    record=record,
                    client=client,
                    model=args.model,
                    prompt_template=prompt_template,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                f_out.write(json.dumps(tagged_record, ensure_ascii=False) + "\n")
                f_out.flush()
    else:
        # Parallel processing
        logger.info(f"Processing in parallel with {args.workers} workers")
        write_lock = Lock()
        
        def process_and_write(record: Dict[str, Any]) -> None:
            """Process single record and write to file (thread-safe)"""
            tagged_record = process_record(
                record=record,
                client=client,
                model=args.model,
                prompt_template=prompt_template,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            
            # Use lock to ensure thread-safe writing
            with write_lock:
                with open(args.output, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(tagged_record, ensure_ascii=False) + "\n")
            
            return tagged_record.get("uuid")
        
        # If not resume mode, first clear or create file
        if not args.resume:
            with open(args.output, "w", encoding="utf-8") as f_out:
                pass
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_and_write, record): record for record in records}
            
            # Use tqdm to show progress
            with tqdm(total=len(records), desc="Tagging nodes") as pbar:
                for future in as_completed(futures):
                    try:
                        uuid = future.result()
                        pbar.update(1)
                    except Exception as e:
                        record = futures[future]
                        logger.error(f"Failed to process record {record.get('uuid')}: {e}")
                        pbar.update(1)
    
    logger.info(f"✅ Tagging completed. Output saved to {args.output}")


if __name__ == "__main__":
    main()
