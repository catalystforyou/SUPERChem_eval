import argparse
import base64
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock, Thread
from typing import Dict, List, Union, Optional

import pandas as pd
import yaml
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

"""
Global variables and configurations
"""
dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

clients = {}
for model_config in config['model_list']:
    model = model_config['model']
    base_url = model_config['base_url']
    api_key = model_config['api_key']
    clients[model] = OpenAI(base_url=base_url, api_key=api_key)

languages = ['en']
prompts = {}
for l in languages:
    with open(os.path.join(dir, f'prompt_{l}.txt'), 'r', encoding='utf-8') as f:
        prompts[l] = f.read()

write_lock = Lock()

"""
Functions
"""
def parse_args() -> argparse.Namespace:
    """
    - input: str, required, the path to the result file.
    - output: str, required, the path to the output file.
    - model: str, required, the name of the model to evaluate.
    - reasoning-effort: str, default None, the reasoning effort for the model's output (low, medium, high).
    - temperature: float, optional, the temperature for the model's output.
    - language: str, required, 'zh' or 'en', the language of the model's input.
    - multimodal: bool, required, whether the input is multimodal or not.
    - pass-k: int, default 1, the number of passes to evaluate.
    - max-retries: int, default 5, the maximum number of retries for a request.
    - timeout: int, default 300, the timeout for each request in seconds.
    - n-procs: int, default 1, the number of processes to use for evaluation.
    - n-threads: int, default 1, the number of threads to use for evaluation.
    - log-level: str, optional, the logging level (DEBUG, INFO, WARNING, ERROR).
    """
    parser = argparse.ArgumentParser(description="Evaluate the results of a model.")
    parser.add_argument('--input', type=str, required=True, help='Path to the result file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to evaluate.')
    parser.add_argument('--reasoning-effort', type=str, default=None, choices=['low', 'medium', 'high'], help='Reasoning effort for the model\'s output.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the model\'s output.')
    parser.add_argument('--language', type=str, required=True, choices=languages, help='Language of the model\'s input.')
    parser.add_argument('--multimodal', type=bool, required=True, help='Whether the input is multimodal or not.')
    parser.add_argument('--pass-k', type=int, default=1, help='Number of passes to evaluate.')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retries for a request.')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout for each request in seconds.')
    parser.add_argument('--n-procs', type=int, default=1, help='Number of processes to use for evaluation.')
    parser.add_argument('--n-threads', type=int, default=1, help='Number of threads to use for evaluation.')
    return parser.parse_args()

def load_data(input_path: str) -> pd.DataFrame:
    """
    Load the input and output data from the specified path. Support .parquet and .jsonl format.
    """
    logger.info(f"Loading data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        exit(1)
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.jsonl'):
        df = pd.read_json(input_path, lines=True)
    else:
        logger.error(f"Unsupported file format: {input_path}. Only .parquet and .jsonl are supported.")
        exit(1)
    return df

def check_incomplete(input_df: pd.DataFrame, output_df: pd.DataFrame, pass_k: int) -> pd.DataFrame:
    """
    Check for incomplete evaluations in the output DataFrame.
    Args:
        input_df: The input DataFrame containing the questions.
        output_df: The output DataFrame containing the model's responses.
        pass_k: The number of passes to evaluate.
    Returns:
        A DataFrame containing the incomplete evaluations.
    """
    # Check if output_df is empty
    if output_df.empty:
        # Repeat all input rows
        incomplete_df = input_df.loc[input_df.index.repeat(pass_k)]
    else:
        # Count successful responses for each uuid
        completed_counts = output_df[output_df['status'] == True].groupby('uuid').size()
        # Calculate remaining needed for each uuid in input_df
        remaining_counts = input_df['uuid'].map(lambda x: pass_k - completed_counts.get(x, 0))
        # Filter and repeat rows based on remaining counts
        mask = remaining_counts > 0
        incomplete_df = input_df[mask].loc[input_df[mask].index.repeat(remaining_counts[mask])]
    logger.info(f"Found {len(incomplete_df)} incomplete evaluations.")

    return incomplete_df.reset_index(drop=True)

def generate_message(input_line: pd.Series, language: str, multimodal: bool) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    """
    Generate the message for the model based on the input line and language.
    Args:
        input_line: A single row from the DataFrame containing the question and options.
        language: The language of the input (either 'zh' or 'en').
        multimodal: Whether the input is multimodal or not.
    Returns:
        A list of dictionaries representing the message to be sent to the model.
    """
    # Choose the appropriate prompt based on the language
    prompt = prompts[language]

    # Generate initial prompt
    question = input_line[f'question_{language}']
    options_dict = input_line[f'options_{language}']
    options = ''
    for key in sorted(options_dict.keys()):
        if options_dict[key] is not None:
            options += f"{key}: {options_dict[key]}\n"
        else:
            break
    prompt = prompt.replace('{question}', question)
    prompt = prompt.replace('{options}', options)

    # Extract <MultiModal> tags
    tag_pattern = r'<MultiModal>(.*?)</MultiModal>'

    if not multimodal:
        def replace_func(match):
            content = match.group(1)
            link_match = re.match(r'!?\s*\[(.+)\]\s*\([^)]+\)', content, flags=re.MULTILINE | re.DOTALL)
            if link_match:
                return f'<MultiModal>{link_match.group(1)}</MultiModal>'
            return match.group(0)
        message = re.sub(tag_pattern, replace_func, prompt, flags=re.IGNORECASE | re.DOTALL)
    else:
        question_images = input_line['question_images'] or {}
        options_images = input_line['options_images'] or {}
        images = {**question_images, **options_images}

        message = []
        last_end = 0
        
        for tag_match in re.finditer(tag_pattern, prompt, re.IGNORECASE | re.DOTALL):
            # Add text before the tag
            if tag_match.start() > last_end:
                message.append({'type': 'text', 'text': prompt[last_end:tag_match.start()]})

            # Parse tag content
            tag_content = tag_match.group(1)
            link_match = re.match(r'!?\s*\[(.+)\]\s*\(([^)]+)\)', tag_content, flags=re.MULTILINE | re.DOTALL)
            
            if link_match:
                image_url = link_match.group(2)
                image_bytes = images.get(image_url)
                
                if image_bytes is None:
                    raise FileNotFoundError(f"Image file not found: {image_url}")
                
                image_data = base64.b64encode(image_bytes).decode('utf-8')
                
                if image_bytes.startswith(b'\xff\xd8\xff'):
                    mimetype = 'jpeg'
                elif image_bytes.startswith(b'\x89PNG'):
                    mimetype = 'png'
                else:
                    raise ValueError(f"Unsupported image format for {image_url}")
                
                message.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/{mimetype};base64,{image_data}',
                        'detail': 'auto',
                    }
                })
            else:
                # If not matching format, treat whole tag as text
                # message.append({'type': 'text', 'text': tag_match.group(0)})
                raise ValueError(f"No link found in multimodal tag: {tag_content}")

            last_end = tag_match.end()
        
        # Add remaining text
        if last_end < len(prompt):
            message.append({'type': 'text', 'text': prompt[last_end:]})

    messages = [{'role': 'user', 'content': message}]
    return messages

def parse_response(response: str) -> Dict[str, str]:
    """
    Parse the model's response to extract the reasoning content and answer.
    Args:
        response: The response string from the model.
    Returns:
        A dictionary containing the reasoning content and answer.
    """
    keys = ['reason', 'answer']

    # JSON block is wrapped in ```json ```
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, response)
    if match:
        json_str = match.group(1)
    else:
        # If no JSON block found, find the largest JSON-like structure
        json_str = response
        json_str = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_str:
            json_str = json_str.group(0)
        else:
            raise ValueError("No valid JSON structure found in the response.")
    # First, try to parse as JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Then try rule-based parsing
        logger.warning("Failed to parse JSON. Attempting rule-based parsing.")
        key_pattern = r'[,\n\s]*["\'](' + '|'.join(re.escape(k) for k in keys) + r')["\']:'
        matches = list(re.finditer(key_pattern, json_str))
        parsed = {}
        for i, match in enumerate(matches):
            key = match.group(1)
            value_start = match.end()
            value_end = matches[i + 1].start() if i < len(matches) - 1 else json_str.rfind('}')
            value = json_str[value_start:value_end].strip(' \n\r,')
            if (value.startswith('"') and value.endswith('"')) or \
                (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            parsed[key] = value
    return 
    
def parse_mcp_response(response: str) -> Dict[str, str]:
    """
    Parse the model's response to extract the reasoning content and answer.
    The answer is expected to be in the format $\boxed{<answer>}$.
    Args:
        response: The response string from the model.
    Returns:
        A dictionary containing the reasoning content and answer.
    """
    pattern = r'\\boxed\{(.+?)\}'
    matches = list(re.finditer(pattern, response, re.DOTALL))

    if matches:
        match = matches[-1]
        answer = match.group(1).strip()
        answer = answer.replace('\\', '').replace('{', '').replace('}', '').replace('text', '').replace('math', '').replace('bf', '').replace('rm', '').strip()
        reason = response
        
        if not reason:
            reason = "No reasoning provided."
            
        return {
            'reason': reason,
            'answer': answer
        }
    else:
        raise ValueError("Could not find the answer in the required $\\boxed{...}$ format.")

def save_result(output_path: str, result: dict):
    """
    Save an additional result to the tail of the output .jsonl file.
    """
    assert output_path.endswith('.jsonl'), "Output file must be in .jsonl format."
    with write_lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            json_str = json.dumps(result, ensure_ascii=False)
            f.write(json_str + '\n')

def handle_request(input_line: pd.Series, model: str, reasoning_effort: Optional[str], temperature: float, language: str, multimodal: bool, timeout: int):
    """
    Send a single request to the model and return the response.
    (MODIFIED to always use streaming for robustness)
    """
    client = clients[model]
    messages = generate_message(input_line, language, multimodal)
    
    # --- Logic is now simplified: always stream ---
    parameters = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'timeout': timeout,
        'stream': True, # Always set to True
        'max_tokens': 32768,
        'stream_options': {"include_usage": True},
    }
    if model == 'gpt-4o-2024-11-20':
        parameters['max_tokens'] = 16384
    if reasoning_effort in ['low', 'medium', 'high']:
        parameters['reasoning_effort'] = reasoning_effort
    
    response = client.chat.completions.create(**parameters)
    
    # --- Process the stream ---
    reasoning = ''
    content = ''
    finish_reason = None
    prompt_usage = None
    completion_usage = None

    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Collect reasoning content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning += delta.reasoning_content
            # Collect main content
            if hasattr(delta, 'content') and delta.content:
                content += delta.content
            # Get the finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        
        # The last chunk of the stream contains the usage data
        if hasattr(chunk, 'usage') and chunk.usage:
            prompt_usage = chunk.usage.prompt_tokens
            completion_usage = chunk.usage.completion_tokens

    # Fallback for models that mix reasoning in content with <think> tags
    if not reasoning and '<think>' in content:
        matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
        if matches:
            reasoning = '\n\n---\n\n'.join(matches)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Check if the model finished correctly
    if finish_reason != 'stop':
        if finish_reason is None:
            logger.warning(f"Model {model} for input {input_line['uuid']} did not return a finish reason. Stream may have been interrupted.")
        else:
            logger.error(f"Model {model} did not finish correctly for input {input_line['uuid']}. Finish reason: {finish_reason}")
            # You might still want to try and process the partial content, or raise an error
            # For robustness, we'll let it pass and let the parse_response function handle it
            # raise ValueError(f"Model did not finish correctly: {finish_reason}")
            
    # In case the stream is cut off before usage stats are sent
    if prompt_usage is None:
        prompt_usage = 0
    if completion_usage is None:
        completion_usage = 0
            
    return content, reasoning, finish_reason, prompt_usage, completion_usage

def single_task(input_line: pd.Series, model: str, reasoning_effort: Optional[str], temperature: float, language: str, multimodal: bool, max_retries: int, timeout: int) -> Dict[str, Union[str, bool]]:
    """
    Handle a single task by sending a request to the model and processing the response.
    Args:
        input_line: A single row from the DataFrame containing the question and options.
        model: The name of the model to evaluate.
        temperature: The temperature for the model's output.
    """
    retries_count = 0
    time_wait = 5
    time_multiplier = 0.5  # Increase wait time by 0.5 seconds for each retry
    while retries_count < max_retries:
        try:
            full_response = handle_request(input_line, model, reasoning_effort, temperature, language, multimodal, timeout)
            # print(full_response)
            content, reasoning, finish_reason, prompt_usage, completion_usage = full_response
            parsed_response = parse_mcp_response(content)
            answer = parsed_response['answer'].strip()  # str
            ref_answer = input_line[f'answer_{language}']   # list[str]
            question_type = input_line['question_type']
            if question_type == 'multiple_choice':
                # There might be multiple selected answers
                # The answer and ref_answer should be totally equal
                score = 1 if len(answer) == len(ref_answer) and all(a.upper() in ref_answer for a in answer) else 0
            elif question_type == 'fill_blank':
                score = 1 if answer.lower() == ref_answer[0].lower() else 0
            else:
                logger.error(f"Unsupported question type: {question_type} for input {input_line['uuid']}")
                raise ValueError(f"Unsupported question type: {question_type}")
            return {
                'uuid': input_line['uuid'],
                'score': score,
                'llm_answer': parsed_response['answer'],
                'llm_output': parsed_response['reason'],
                'llm_reasoning': reasoning,
                'finish_reason': finish_reason,
                'prompt_usage': prompt_usage,
                'completion_usage': completion_usage,
                'model': model,
                'reasoning_effort': reasoning_effort,
                'temperature': temperature,
                'language': language,
                'multimodal': multimodal,
                'status': True,
            }
        except Exception as e:
            retries_count += 1
            logger.error(f"Error processing {input_line['uuid']} with model {model}: {e}. Retrying {retries_count}/{max_retries}...")
            if retries_count >= max_retries:
                logger.error(f"Max retries reached for {input_line['uuid']} with model {model}.")
                return {
                    'uuid': input_line['uuid'],
                    'model': model,
                    'reasoning_effort': reasoning_effort,
                    'temperature': temperature,
                    'language': language,
                    'multimodal': multimodal,
                    'status': False,
                    'error': str(e)
                }
            time.sleep(time_wait)
            time_wait += time_multiplier * retries_count

def worker(task_queue, args, process_id, thread_id, progress_counter, progress_lock):
    """Worker function for processing tasks"""
    logger.info(f"Process {process_id}, Thread {thread_id} started")
    output_path, model, reasoning_effort, temperature, language, multimodal, max_retries, timeout = args
    while True:
        try:
            input_line = task_queue.get(timeout=1)
            if input_line is None:  # Poison pill
                break
            result = single_task(input_line, model, reasoning_effort, temperature, language, multimodal, max_retries, timeout)
            save_result(output_path, result)
            with progress_lock:
                progress_counter.value += 1
        except Exception as e:
            logger.error(f"Error in process {process_id}, thread {thread_id}: {e}")
            break
    logger.info(f"Process {process_id}, Thread {thread_id} finished")

def process_worker(task_queue, args, n_threads, process_id, progress_counter, progress_lock):
    """Process worker that spawns multiple threads"""
    logger.info(f"Process {process_id} started with {n_threads} threads")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker, task_queue, args, process_id, thread_id, progress_counter, progress_lock) 
                   for thread_id in range(n_threads)]
        for future in futures:
            future.result()
    logger.info(f"Process {process_id} finished")

def main():
    manager = mp.Manager()
    task_queue = manager.Queue()

    args = parse_args()
    logger.info(f"Loading configuration from {dir}")
    logger.info(f"Clients initialized for models: {list(clients.keys())}")
    logger.info(f"Starting evaluation with model: {args.model}, n_procs: {args.n_procs}, n_threads: {args.n_threads}")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"[Prompt]\n\n{prompts[args.language]}")
    
    # Load data
    input_df = load_data(args.input)
    
    # Check if output file exists
    if os.path.exists(args.output):
        output_df = pd.read_json(args.output, lines=True)
    else:
        output_df = pd.DataFrame()
    
    # Check incomplete tasks
    incomplete_df = check_incomplete(input_df, output_df, args.pass_k)

    if len(incomplete_df) == 0:
        logger.info("No incomplete tasks found. Exiting.")
        return
    
    # Initialize progress bar
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()

    def update_progress_bar(pbar, counter, total):
        while counter.value < total:
            pbar.n = counter.value
            pbar.refresh()
            time.sleep(0.1)
        pbar.n = total
        pbar.refresh()
    
    pbar = tqdm(total=len(incomplete_df), desc="Processing tasks")

    progress_thread = Thread(target=update_progress_bar, args=(pbar, progress_counter, len(incomplete_df)))
    progress_thread.start()
    
    # Fill task queue
    for _, row in incomplete_df.iterrows():
        task_queue.put(row)
    
    # Add poison pills
    for _ in range(args.n_procs * args.n_threads):
        task_queue.put(None)
    
    # Create worker arguments
    worker_args = (args.output, args.model, args.reasoning_effort, args.temperature, args.language,
                   args.multimodal, args.max_retries, args.timeout)
    
    # Start multiprocessing
    if args.n_procs == 1:
        process_worker(task_queue, worker_args, args.n_threads, 0, progress_counter, progress_lock)
    else:
        with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
            futures = [executor.submit(process_worker, task_queue, worker_args, args.n_threads, process_id, progress_counter, progress_lock) 
                    for process_id in range(args.n_procs)]
            for future in futures:
                future.result()

    progress_thread.join()
    pbar.close()
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()