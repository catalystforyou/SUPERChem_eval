import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ##############################################################################
# 1. Configuration Area
# ##############################################################################

# SOTA models for target analysis
SOTA_MODELS = [
    "deepseek-v3_1-think-128k", 
    "gpt-5_high", 
    "gemini-2_5-pro"
]

# Pretty names for plotting
PRETTY_NAMES_MAP = {
    "gpt-5_high": "GPT-5 (High)",
    "gemini-2_5-pro": "Gemini 2.5 Pro",
    "deepseek-v3_1-think-128k": "DeepSeek-V3.1-Think",
}

# Directory path where the data files are located
# Please ensure your jsonl files are in these directories
CKPT_TAG_DIR = 'data/' # Checkpoint definition file directory
EVAL_RESULTS_DIR = 'data/' # Model evaluation file directory (assuming in the current directory)


# ##############################################################################
# 2. Helper Functions and Data Loading
# ##############################################################################

def parse_filename_final_v2(filename, known_models):
    """
    V2 parsing function: can correctly handle filenames containing '_EVAL_BY_'.
    """
    basename = os.path.basename(filename)
    search_part = basename.split('_EVAL_BY_')[0]
    sorted_known_models = sorted(known_models, key=len, reverse=True)
    
    found_model = None
    for model in sorted_known_models:
        if model in search_part:
            found_model = model
            break
    if not found_model:
        return None, None
        
    pre_model_part = basename.split(found_model)[0]
    multimodal = '_true_' in pre_model_part
    return found_model, multimodal

def load_checkpoint_tags(directory):
    """
    Load checkpoint definitions for all problems and map them to UUIDs.
    """
    ckpt_tags_map = {}
    files = glob.glob(os.path.join(directory, '*.jsonl'))
    print(f"Found {len(files)} checkpoint tag definition files.")
    
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'uuid' in record and 'checkpoints_analysis' in record:
                        ckpt_tags_map[record['uuid']] = record['checkpoints_analysis']
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Successfully loaded checkpoint tags for {len(ckpt_tags_map)} unique problems.")
    return ckpt_tags_map

# ##############################################################################
# 3. Core Analysis Logic
# ##############################################################################

def analyze_reasoning_breakpoints(eval_files, sota_models, ckpt_tags_map):
    """
    Analyze all failed reasoning paths, find the first breakpoint, and count its ability tags.
    """
    # Filter files for analysis according to the rule: if a model has multimodal results, use only the multimodal ones
    files_to_process = []
    model_runs = defaultdict(dict)
    for f in eval_files:
        model, multimodal = parse_filename_final_v2(f, sota_models)
        if model:
            modality = 'Multimodal' if multimodal else 'Text-Only'
            model_runs[model][modality] = f

    for model in sota_models:
        if 'Multimodal' in model_runs[model]:
            files_to_process.append(model_runs[model]['Multimodal'])
            print(f"Selecting Multimodal results for {model}")
        elif 'Text-Only' in model_runs[model]:
            files_to_process.append(model_runs[model]['Text-Only'])
            print(f"Selecting Text-Only results for {model}")
        else:
            print(f"Warning: No evaluation results found for SOTA model: {model}")

    breakpoint_stats = defaultdict(lambda: defaultdict(int))

    for filepath in files_to_process:
        model_name, _ = parse_filename_final_v2(filepath, sota_models)
        if not model_name:
            continue
            
        print(f"\nProcessing file: {os.path.basename(filepath)}")
        failure_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    uuid = record.get('uuid')
                    total_ckpts = record.get('total_checkpoints', 0)
                    matched_ckpts = record.get('matched_checkpoints', 0)

                    # Condition: must be a failed reasoning case
                    if total_ckpts > 0 and matched_ckpts < total_ckpts:
                        failure_count += 1
                        # Find the first unmatched checkpoint
                        for i, detail in enumerate(record.get('evaluation_details', [])):
                            if not detail.get('is_matched', True):
                                # Found the first breakpoint
                                if uuid in ckpt_tags_map and i < len(ckpt_tags_map[uuid]):
                                    # Get the corresponding ability tags from the definition file
                                    breakpoint_ckpt = ckpt_tags_map[uuid][i]
                                    ability_tags = breakpoint_ckpt.get('ability_tags', [])
                                    for tag in ability_tags:
                                        breakpoint_stats[model_name][tag] += 1
                                # Break out of the inner loop immediately after finding it, only counting the first breakpoint
                                break
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Found {failure_count} failed reasoning paths for {model_name}.")
        
    return breakpoint_stats

# ##############################################################################
# 4. Plotting Function (Modified)
# ##############################################################################

def plot_breakpoint_distribution(stats):
    """
    Plot the breakpoint statistics as a grouped bar chart, using Arial font and saving as a PDF.
    """
    if not stats:
        print("No breakpoint data to plot.")
        return
        
    # --- New: Set global font to Arial ---
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial'] # Ensure a fallback if Arial is not found
    
    # --- Set line widths ---
    plt.rcParams['axes.linewidth'] = 2.5  # Chart border width
    plt.rcParams['xtick.major.width'] = 2.5 # X-axis tick width
    plt.rcParams['ytick.major.width'] = 2.5 # Y-axis tick width

    # Convert dictionary to Pandas DataFrame for easy plotting
    records = []
    for model, tags in stats.items():
        total_breakpoints = sum(tags.values())
        if total_breakpoints == 0: continue
        for tag, count in tags.items():
            percentage = (count / total_breakpoints) * 100
            records.append({
                'Model': PRETTY_NAMES_MAP.get(model, model),
                'Ability Tag': tag,
                'Percentage': percentage
            })
            
    df = pd.DataFrame(records)
    
    if df.empty:
        print("DataFrame is empty after processing stats. Cannot plot.")
        return

    # --- Modified: Keep only top 10 ---
    tag_order = df.groupby('Ability Tag')['Percentage'].sum().sort_values(ascending=False).index[:10]
    df = df[df['Ability Tag'].isin(tag_order)]

    # --- Start plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    sns.barplot(
        data=df,
        y='Ability Tag',
        x='Percentage',
        hue='Model',
        order=tag_order,
        palette='viridis',
        ax=ax,
        edgecolor='black',
        linewidth=2.5
    )

    # --- Beautify the chart (larger fonts, bold axes) ---
    # ax.set_title('Distribution of First Reasoning Breakpoints for SOTA Models', fontsize=30, pad=20, weight='bold')
    ax.set_ylabel('Ability Tag of First Failed Checkpoint', fontsize=32, labelpad=20, weight='bold')
    ax.set_xlabel('Percentage of First Breakpoints (%)', fontsize=32, labelpad=20, weight='bold')

    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=24, width=2.5, length=8)
    ax.tick_params(axis='y', width=2.5, length=8)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    
    # Bold borders
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5, linewidth=1.5)

    # Get legend and set font
    legend = ax.legend(title='Model', fontsize=22, title_fontsize=24, loc='lower right', frameon=True, framealpha=0.9, edgecolor='black')
    legend.get_frame().set_linewidth(2.0)
    plt.setp(legend.get_texts(), fontname='Arial')
    plt.setp(legend.get_title(), fontname='Arial', weight='bold')

    # sns.despine(top=True, right=True)
    # Reset left and bottom spine linewidths (despine might affect them)
    # ax.spines['left'].set_linewidth(2.5)
    # ax.spines['bottom'].set_linewidth(2.5)

    plt.tight_layout()
    
    # --- Modified: Save as PDF format ---
    output_filename = "results/first_breakpoint_analysis.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight') # Use format='pdf'
    print(f"\nPlot saved successfully as '{output_filename}'")
    plt.show()


# ##############################################################################
# 5. Main Program Entry
# ##############################################################################

if __name__ == "__main__":
    # Step 1: Load the standard answer Checkpoint Tags for all problems
    checkpoint_tags = load_checkpoint_tags(CKPT_TAG_DIR)

    if not checkpoint_tags:
        print("Could not load checkpoint tags. Aborting analysis.")
    else:
        # Step 2: Find all model evaluation files
        all_eval_files = glob.glob(os.path.join(EVAL_RESULTS_DIR, '*_EVAL_BY_*.jsonl'))
        
        # Step 3: Perform breakpoint analysis
        breakpoint_results = analyze_reasoning_breakpoints(all_eval_files, SOTA_MODELS, checkpoint_tags)

        # Step 4: Print the statistical results
        print("\n--- First Breakpoint Statistics (Raw Counts) ---")
        for model, tags in breakpoint_results.items():
            print(f"\nModel: {PRETTY_NAMES_MAP.get(model, model)}")
            sorted_tags = sorted(tags.items(), key=lambda item: item[1], reverse=True)
            for tag, count in sorted_tags:
                print(f"  {tag}: {count}")
        
        # Step 5: Plot and save the chart
        plot_breakpoint_distribution(breakpoint_results)