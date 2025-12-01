import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ##############################################################################
# 1. Configuration Area (Copied from your reference code)
# ##############################################################################
KNOWN_MODELS = [
    "qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3_1-think-128k", "deepseek-v3_1-128k",
    "gemini-2_5-pro", "o3-2025-04-16",
    "gpt-5_high", "gpt-5_medium", "gpt-5_low",
    "gpt-4_1", "gpt-4o", "gpt-5_minimal"
]

PRETTY_NAMES_MAP = {
    "gpt-5_high": "GPT-5 (High)", "gpt-5_medium": "GPT-5 (Medium)",
    "gpt-5_low": "GPT-5 (Low)", "gemini-2_5-pro": "Gemini 2.5 Pro",
    "o3-2025-04-16": "o3 (High)", "deepseek-v3_1-think-128k": "DeepSeek-V3.1-Think",
    "deepseek-v3_1-128k": "DeepSeek-V3.1", "qwen3-235b-a22b-thinking-2507": "Qwen3-235B-Think",
    "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-Instruct",
    "gpt-4_1": "GPT-4.1", "gpt-4o": "GPT-4o", "gpt-5_minimal": "GPT-5 (Minimal)"
}

# Models designated for this specific SOTA analysis
SOTA_MODELS = [
    "gpt-5_high", "gemini-2_5-pro", "deepseek-v3_1-think-128k",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507"
]

SPLIT_MAP_FILE = 'data/dataset_split_map.json'
K_VALUES_TO_CALCULATE = [1, 2, 4, 8]

# ##############################################################################
# 2. Data Loading Functions (Adapted from your reference code)
# ##############################################################################

def parse_filename_final(filename, known_models):
    """Parses model name, multimodal status, and pass_k from a filename."""
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
    """Loads and filters scores from .jsonl files based on the split map."""
    try:
        with open(split_map_file, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        valid_uuids = {
            uuid for uuid, data in split_map.items() 
            if data.get('split') in ['release', 'holdout']
        }
        print(f"Loaded split map. Found {len(valid_uuids)} valid questions to analyze.")
    except FileNotFoundError:
        print(f"Error: '{split_map_file}' not found. Aborting.")
        return None
    
    raw_data = defaultdict(lambda: {'scores': defaultdict(list), 'pass_k': 1})
    for filepath in files:
        model_name, multimodal, pass_k = parse_filename_final(filepath, known_models)
        if not model_name: continue
        key = (model_name, multimodal)
        raw_data[key]['pass_k'] = max(raw_data[key]['pass_k'], pass_k)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record['uuid'] in valid_uuids:
                        raw_data[key]['scores'][record['uuid']].append(record['score'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return raw_data

# ##############################################################################
# 3. Functions for pass@k Calculation and Plotting (MODIFIED)
# ##############################################################################

def calculate_pass_k_metrics(raw_data):
    """
    Calculates pass@k for k in {1, 2, 4, 8} for SOTA models.
    Prioritizes multimodal results if available.
    """
    pass_k_results = []
    
    print("\nCalculating pass@k metrics for SOTA models...")
    for model_name in SOTA_MODELS:
        multimodal_key = (model_name, True)
        text_key = (model_name, False)
        
        data_to_use = None
        if multimodal_key in raw_data and raw_data[multimodal_key].get('pass_k', 0) >= 8:
            data_to_use = raw_data[multimodal_key]
        elif text_key in raw_data and raw_data[text_key].get('pass_k', 0) >= 8:
            data_to_use = raw_data[text_key]

        if not data_to_use:
            print(f"  - Skipping '{model_name}': No data found with at least 8 attempts.")
            continue

        scores_dict = data_to_use['scores']
        if not scores_dict:
            print(f"  - Skipping '{model_name}': No valid scores found.")
            continue
            
        scores_matrix = np.array([scores[:8] for scores in scores_dict.values() if len(scores) >= 8])
        num_questions = scores_matrix.shape[0]

        if num_questions == 0:
            print(f"  - Skipping '{model_name}': Not enough questions with 8 attempts.")
            continue
            
        print(f"  + Processing '{model_name}' ({num_questions} questions)...")

        for k in K_VALUES_TO_CALCULATE:
            num_correct_at_k = np.sum(np.any(scores_matrix[:, :k] == 1, axis=1))
            accuracy_at_k = (num_correct_at_k / num_questions) * 100
            
            pass_k_results.append({
                'Model': model_name,
                'k': k,
                'Accuracy': accuracy_at_k
            })
            
    return pd.DataFrame(pass_k_results)


def plot_pass_k_evolution(pass_k_df):
    """
    Generates and saves a line plot of pass@k accuracy vs. number of attempts (k).
    Uses Arial font, increased font sizes, and saves as a PDF.
    """
    if pass_k_df.empty:
        print("Cannot generate plot: The calculated metrics DataFrame is empty.")
        return

    print("\n--- Generating SOTA pass@k Evolution Plot ---")
    
    # --- MODIFICATION: Set global font to Arial ---
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    pass_k_df['Pretty Model'] = pass_k_df['Model'].map(PRETTY_NAMES_MAP)
    
    sort_order = (
        pass_k_df[pass_k_df['k'] == max(K_VALUES_TO_CALCULATE)]
        .sort_values('Accuracy', ascending=False)['Pretty Model']
        .tolist()
    )

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10)) # Increased height slightly for better spacing

    sns.lineplot(
        data=pass_k_df,
        x='k',
        y='Accuracy',
        hue='Pretty Model',
        hue_order=sort_order,
        style='Pretty Model',
        markers=True,
        dashes=False,
        lw=4.0, # Made lines thicker
        markersize=14, # Made markers larger
        ax=ax
    )
        
    # --- MODIFICATION: Formatting and Styling with larger fonts ---
    # ax.set_title("SOTA Model Performance with Multiple Attempts on CCMEBench", fontsize=26, pad=25, weight='bold')
    ax.set_xlabel("Number of Attempts (k)", fontsize=28, labelpad=15, weight='bold')
    ax.set_ylabel("pass@k Accuracy (%)", fontsize=28, labelpad=15, weight='bold')
    
    ax.set_xticks(K_VALUES_TO_CALCULATE)
    ax.set_xticklabels(K_VALUES_TO_CALCULATE, fontsize=24, weight='bold')
    
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    ax.tick_params(axis='y', labelsize=24)
    
    # Thicken ticks and spines
    ax.tick_params(width=2.5, length=8)
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color('black')
    
    # Get legend handle and apply larger fonts
    legend = ax.legend(
        title='Model', 
        fontsize=20, 
        title_fontsize=22,
        frameon=True,       # Enable frame
        framealpha=1.0,     # Opaque background to cover lines
        edgecolor='black',  # Black border color
        facecolor='white'   # White background color
    )
    legend.get_frame().set_linewidth(1.5) # Thicker border for the "card" look
    plt.setp(legend.get_texts(), fontname='Arial') # Explicitly set legend text font
    plt.setp(legend.get_title(), fontname='Arial', weight='bold') # Explicitly set legend title font
    
    plt.tight_layout()
    
    # --- MODIFICATION: Save as PDF ---
    output_filename = "results/sota_pass_k_evolution.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved successfully to '{output_filename}'")


# ##############################################################################
# 4. Main Execution Block
# ##############################################################################

if __name__ == "__main__":
    RESULTS_DIR = 'data/' 
    
    all_json_files = glob.glob(os.path.join(RESULTS_DIR, '*.jsonl'))
    if not all_json_files:
        print("Error: No .jsonl files found in the current directory.")
    else:
        print(f"Found {len(all_json_files)} result files to process.")
        
        aggregated_data = process_data_files(all_json_files, KNOWN_MODELS)
        
        if aggregated_data:
            pass_k_metrics_df = calculate_pass_k_metrics(aggregated_data)
            
            plot_pass_k_evolution(pass_k_metrics_df)
            
            print("\n--- pass@k Metrics Summary ---")
            if not pass_k_metrics_df.empty:
                print(pass_k_metrics_df.pivot_table(index='Model', columns='k', values='Accuracy').to_string(float_format="%.1f"))
            else:
                print("No pass@k metrics were calculated.")
        else:
            print("Data processing failed.")