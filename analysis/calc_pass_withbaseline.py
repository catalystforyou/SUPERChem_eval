import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np

# ##############################################################################
# 1. Configuration Area
# ##############################################################################
KNOWN_MODELS = [
    "qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3_1-think-128k", "deepseek-v3_1-128k",
    "gemini-2_5-pro", "o3-2025-04-16",
    "gpt-5_high", "gpt-5_medium", "gpt-5_low",
    "gpt-4_1", "gpt-4o", "gpt-5_minimal"
]

SOTA_MODELS = [
    "gpt-5_high", "gemini-2_5-pro", "deepseek-v3_1-think-128k",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507"
]

SPLIT_MAP_FILE = 'data/dataset_split_map.json'
# ##############################################################################

def calculate_human_baseline(csv_filepath, valid_uuids):
    """
    Read human evaluation data from a CSV file and calculate accuracy.
    A question is considered correct if at least one person answered it correctly.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"\nWarning: Human baseline file not found at '{csv_filepath}'. Skipping baseline calculation.")
        return None

    # Ensure calculation is only on the same set of valid questions as the models
    df_filtered = df[df['uuid'].isin(valid_uuids)]
    if df_filtered.empty:
        print("Warning: No matching human baseline data found for the valid question set.")
        return None

    # Group by uuid and check if any score is 1
    correct_by_uuid = df_filtered.groupby('uuid')['score'].max()
    
    num_correct = correct_by_uuid.sum()
    total_questions = len(correct_by_uuid)

    accuracy = (num_correct / total_questions) * 100
    print(f"Human Baseline calculated: {accuracy:.1f}% ({num_correct}/{total_questions} questions correct)")
    return accuracy

def parse_filename_final(filename, known_models):
    """
    Parse model name, multimodal status, and pass_k from a filename.
    """
    basename = os.path.basename(filename)
    found_model = next((model for model in known_models if model in basename), None)
    if not found_model: return None, None, None
    try:
        pre_model_part = basename.split(found_model)[0]
        multimodal = '_true_' in pre_model_part
        pass_k_str = basename.rsplit('.', 1)[0].rsplit('_', 1)[-1]
        pass_k = int(pass_k_str)
        return found_model, multimodal, pass_k
    except (IndexError, ValueError): return None, None, None

def process_data_files(files, known_models, split_map_file=SPLIT_MAP_FILE):
    """
    Process all result files for text-only inputs.
    """
    try:
        with open(split_map_file, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        valid_uuids = {
            uuid for uuid, data in split_map.items() 
            if data.get('split') in ['release', 'holdout']
        }
        print(f"Loaded split map. Found {len(valid_uuids)} non-easy questions to analyze.")
    except FileNotFoundError:
        print(f"Error: '{split_map_file}' not found. Aborting.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{split_map_file}'. Aborting.")
        return None, None
    
    raw_data = defaultdict(lambda: {'scores': defaultdict(list), 'pass_k': 1})
    for filepath in files:
        model_name, multimodal, pass_k = parse_filename_final(filepath, known_models)
        if not model_name or multimodal: 
            continue # Skip if no model name or if it's a multimodal result
        
        raw_data[model_name]['pass_k'] = max(raw_data[model_name]['pass_k'], pass_k)
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        np.random.seed(0)
        np.random.shuffle(lines)
        for line in lines:
            try:
                record = json.loads(line)
                if record['uuid'] in valid_uuids:
                    raw_data[model_name]['scores'][record['uuid']].append(record['score'])
            except (json.JSONDecodeError, KeyError):
                continue
    return raw_data, valid_uuids

def calculate_metrics_v2(raw_data):
    """
    Calculate pass@1, pass@8, and reliability metrics from raw data.
    """
    metrics = []
    for model_name, data in raw_data.items():
        scores_dict = data['scores']
        if not scores_dict: continue
        total_questions = len(scores_dict)
        
        # Calculate pass@1 (First Trial)
        first_trial_correct = sum(1 for scores in scores_dict.values() if scores and scores[0] == 1)
        pass1_first_trial_acc = (first_trial_correct / total_questions) * 100
        metrics.append({
            'Model': model_name, 'Metric': 'pass@1 (First Trial)', 
            'Accuracy': pass1_first_trial_acc, 'StdDev': 0.0
        })
        
        # Calculate pass@1 (Mean Reliability) and pass@8 for SOTA models
        if model_name in SOTA_MODELS and data['pass_k'] >= 8:
            scores_matrix = np.array([scores[:8] for scores in scores_dict.values() if len(scores) >= 8 and all(s is not None for s in scores[:8])])
            if scores_matrix.shape[0] > 0:
                # Mean Reliability
                accuracies_per_trial = np.mean(scores_matrix, axis=0) * 100
                pass1_acc_mean = np.mean(accuracies_per_trial)
                pass1_acc_std = np.std(accuracies_per_trial)
                metrics.append({
                    'Model': model_name, 'Metric': 'pass@1 (Mean Reliability)', 
                    'Accuracy': pass1_acc_mean, 'StdDev': pass1_acc_std
                })
                
                # pass@8
                pass8_correct = np.sum(np.any(scores_matrix == 1, axis=1))
                pass8_acc = (pass8_correct / scores_matrix.shape[0]) * 100
                metrics.append({
                    'Model': model_name, 'Metric': 'pass@8', 
                    'Accuracy': pass8_acc, 'StdDev': 0.0
                })
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    RESULTS_DIR = 'data/' 
    
    all_json_files = glob.glob(os.path.join(RESULTS_DIR, '*.jsonl'))
    if not all_json_files:
        print("No .jsonl files found in the data directory.")
    else:
        print(f"Found {len(all_json_files)} files to process.")
        aggregated_data, valid_uuids = process_data_files(all_json_files, KNOWN_MODELS)
        
        if aggregated_data:
            human_baseline_file = 'data/20251015_baseline.csv'
            print("\n--- Analyzing Human Performance ---")
            human_accuracy = calculate_human_baseline(human_baseline_file, valid_uuids)

            print("\nCalculating model metrics...")
            metrics_df = calculate_metrics_v2(aggregated_data)
            
            if not metrics_df.empty:
                print("\n--- Text-Only Model Performance Summary ---")
                summary_table = metrics_df.pivot_table(
                    index='Model', columns='Metric', values=['Accuracy', 'StdDev']
                ).sort_index()
                
                # Reorder columns for better readability
                desired_order = [
                    ('Accuracy', 'pass@1 (First Trial)'),
                    ('Accuracy', 'pass@1 (Mean Reliability)'),
                    ('StdDev', 'pass@1 (Mean Reliability)'),
                    ('Accuracy', 'pass@8')
                ]
                # Filter out columns that don't exist in the summary_table
                existing_columns = [col for col in desired_order if col in summary_table.columns]
                summary_table = summary_table[existing_columns]

                print(summary_table.to_string(float_format="%.1f"))
            else:
                print("Metric calculation resulted in an empty DataFrame.")
        else:
            print("Data processing failed.")