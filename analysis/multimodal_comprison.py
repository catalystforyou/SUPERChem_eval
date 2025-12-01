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

# --- File Paths ---
QUESTIONS_PATH = 'data/20251014164938_questions.parquet'
SPLIT_MAP_FILE = 'data/dataset_split_map.json'
RESULTS_DIR = 'data/' # Assume result files are in the current directory

# --- Model Name Mapping ---
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

# ##############################################################################
# 2. Helper Functions (Keep as is)
# ##############################################################################

def get_multimodal_release_uuids(questions_path: str, split_map_path: str) -> set:
    """
    Identifies the UUIDs of all questions containing images in the 'release' dataset.
    """
    # 1. Load UUIDs for the 'release' set
    try:
        with open(split_map_path, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        release_uuids = {
            uuid for uuid, data in split_map.items()
            if data.get('split') == 'release'
        }
        print(f"Loaded split map. Found {len(release_uuids)} questions in the 'release' set.")
    except FileNotFoundError:
        print(f"Error: Split map file not found at '{split_map_path}'.")
        return set()

    # 2. Load question data and identify questions with images
    try:
        questions_df = pd.read_parquet(questions_path)
        # If the column is not a missing value (NaN/None), it is considered to have an image
        has_question_image = questions_df['question_images'].notna()
        has_options_image = questions_df['options_images'].notna()
        
        multimodal_df = questions_df[has_question_image | has_options_image] # Using OR (|) logic is more accurate
        multimodal_uuids = set(multimodal_df['uuid'])
        print(f"Loaded questions data. Found {len(multimodal_uuids)} questions with images in the full dataset.")
    except FileNotFoundError:
        print(f"Error: Questions file not found at '{questions_path}'.")
        return set()
    except Exception as e:
        print(f"An error occurred while reading the parquet file: {e}")
        return set()

    # 3. Return the intersection of the two
    multimodal_release_uuids = release_uuids.intersection(multimodal_uuids)
    print(f"Analysis will be performed on the {len(multimodal_release_uuids)} questions that are in the 'release' set AND have images.")
    
    return multimodal_release_uuids


def parse_filename(filename: str, known_models: list) -> tuple:
    """Parses the model name and input type (multimodal/text-only) from the filename."""
    basename = os.path.basename(filename)
    # Prioritize matching longer model names to avoid "gpt-4o" being incorrectly matched as "gpt-4"
    sorted_known_models = sorted(known_models, key=len, reverse=True)
    found_model = next((model for model in sorted_known_models if model in basename), None)
    if not found_model:
        return None, None
    
    pre_model_part = basename.split(found_model)[0]
    is_multimodal = '_true_' in pre_model_part
    input_type = 'Multimodal' if is_multimodal else 'Text-Only'
    
    return found_model, input_type

def process_model_results(results_dir: str, target_uuids: set, known_models: list) -> dict:
    """
    Loads model result files and processes scores only for the target UUID subset.
    """
    all_json_files = glob.glob(os.path.join(results_dir, '*.jsonl'))
    
    # Structure: results[model_name][input_type] = [scores_list]
    results = defaultdict(lambda: defaultdict(list))

    for filepath in all_json_files:
        model_name, input_type = parse_filename(filepath, known_models)
        if not model_name:
            continue
        
        # Use a dictionary to ensure each UUID is recorded only once
        recorded_uuid = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        np.random.seed(1025)
        np.random.shuffle(lines)
        for line in lines:
            try:
                record = json.loads(line)
                if record.get('uuid') in target_uuids and record.get('uuid') not in recorded_uuid:
                    recorded_uuid.add(record.get('uuid'))
                    # We only care about pass@1, so we record the score directly
                    results[model_name][input_type].append(record.get('score', 0))
            except (json.JSONDecodeError, KeyError):
                continue
    return results

# ##############################################################################
# 3. Main Execution Logic
# ##############################################################################

if __name__ == "__main__":
    # 1. Determine the set of question UUIDs to be analyzed
    multimodal_subset_uuids = get_multimodal_release_uuids(QUESTIONS_PATH, SPLIT_MAP_FILE)

    if not multimodal_subset_uuids:
        print("\nCould not determine the subset of questions to analyze. Aborting.")
    else:
        # 2. Process the result files for all models
        raw_results = process_model_results(RESULTS_DIR, multimodal_subset_uuids, KNOWN_MODELS)
        
        # 3. Identify models that have both Text-Only and Multimodal results
        models_to_compare = []
        for model, data in raw_results.items():
            if 'Text-Only' in data and 'Multimodal' in data:
                models_to_compare.append(model)
        
        # 4. Calculate and organize the accuracy of these models
        analysis_data = []
        total_questions = len(multimodal_subset_uuids)

        for model in models_to_compare:
            for input_type in ['Multimodal', 'Text-Only']:
                scores = raw_results[model][input_type]
                num_correct = sum(scores)
                # Ensure the denominator is the size of the subset, even if the model missed some questions
                accuracy = (num_correct / total_questions) * 100 if total_questions > 0 else 0
                
                analysis_data.append({
                    'Model': PRETTY_NAMES_MAP.get(model, model),
                    'Input Type': input_type,
                    'Correct': num_correct,
                    'Total': total_questions,
                    'Accuracy (%)': accuracy
                })
        
        if not analysis_data:
            print("\nNo models found with both Text-Only and Multimodal results for the specified subset.")
        else:
            # 5. Print the results table
            results_df = pd.DataFrame(analysis_data)
            print(f"\n--- Performance on Multimodal-Essential Subset ({total_questions} questions) ---")
            print(results_df[['Model', 'Input Type', 'Correct', 'Total', 'Accuracy (%)']].to_string(index=False))

            # 6. Plot a comparison bar chart (modified)
            # Set global plotting style, mimicking the provided example
            plt.rcParams.update({
                'font.size': 18,
                'axes.linewidth': 2,
                'xtick.major.width': 2,
                'ytick.major.width': 2,
                'xtick.major.size': 6,
                'ytick.major.size': 6,
            })
            
            # --- CHANGE 3: Set global font to Arial ---
            try:
                plt.rcParams['font.family'] = 'Arial'
                print("\nSetting plot font to Arial.")
            except RuntimeError:
                print("\nArial font not found. Using default font.")

            fig, ax = plt.subplots(figsize=(12, 8)) # Slightly larger canvas
            
            # --- CHANGE 2: Determine the order of the bars ---
            # Sort by Multimodal performance from high to low
            model_order = results_df[results_df['Input Type'] == 'Multimodal'].sort_values(
                'Accuracy (%)', ascending=False
            )['Model'].tolist()

            colors = {'Text-Only': '#014f86', 'Multimodal': '#ff6a00'}
            
            # Use the order parameter in the plotting function, add borders
            sns.barplot(data=results_df, x='Model', y='Accuracy (%)', hue='Input Type', 
                        palette=colors, ax=ax, order=model_order,
                        edgecolor='black', linewidth=1.5)
            
            # Add value labels on the bar chart
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}', 
                                (p.get_x() + p.get_width() / 2., height), 
                                ha='center', va='bottom', xytext=(0, 8), 
                                textcoords='offset points', fontsize=18, color='black')

            # Set axis labels and title
            ax.set_xlabel(None)
            ax.set_ylabel('pass@1 Accuracy (%)', fontsize=24, labelpad=15, color='black')
            
            # Set axis limits and format
            ax.set_ylim(0, max(50, results_df['Accuracy (%)'].max() * 1.15))
            ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
            
            # Set tick labels font
            plt.xticks(rotation=45, ha='right', fontsize=20, color='black')
            plt.yticks(fontsize=20, color='black')
            
            # Set legend
            ax.legend(title='Input Type', fontsize=18, title_fontsize=20, 
                      frameon=True, edgecolor='black', fancybox=False)
            
            # Ensure borders on all sides (Top and Right spines)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)

            # Grid settings (Keep only horizontal grid, below the plot)
            ax.grid(False) # Turn off default grid first
            ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
            ax.set_axisbelow(True)

            plt.tight_layout()
            
            # --- CHANGE 1: Save as PNG and PDF simultaneously ---
            png_output_filename = 'results/textonly_subset_comparison.png'
            pdf_output_filename = 'results/textonly_subset_comparison.pdf'
            
            plt.savefig(png_output_filename, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved as '{png_output_filename}'")
            
            plt.savefig(pdf_output_filename, bbox_inches='tight')
            print(f"Comparison plot also saved as '{pdf_output_filename}'")

            plt.close(fig)