import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ##############################################################################
# 1. Configuration Area
# ##############################################################################

# This list should contain all models you want to include in the analysis.
# The script will automatically select the correct input type (Multimodal > Text-Only).
KNOWN_MODELS = [
    "qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3_1-think-128k", "deepseek-v3_1-128k",
    "gemini-2_5-pro", "o3-2025-04-16",
    "gpt-5_high", "gpt-5_medium", "gpt-5_low",
    "gpt-4_1", "gpt-4o", "gpt-5_minimal"
]

# Pretty names for plotting
PRETTY_NAMES_MAP = {
    "gpt-5_high": "GPT-5 (High)", "gpt-5_medium": "GPT-5 (Medium)",
    "gpt-5_low": "GPT-5 (Low)", "gemini-2_5-pro": "Gemini 2.5 Pro",
    "o3-2025-04-16": "o3 (High)", "deepseek-v3_1-think-128k": "DeepSeek-V3.1-Think",
    "deepseek-v3_1-128k": "DeepSeek-V3.1", "qwen3-235b-a22b-thinking-2507": "Qwen3-235B-Think",
    "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-Instruct",
    "gpt-4_1": "GPT-4.1", "gpt-4o": "GPT-4o", "gpt-5_minimal": "GPT-5 (Minimal)"
}

SPLIT_MAP_FILE = 'data/dataset_split_map.json'
SAMPLE_SIZES = [1, 2, 5, 10, 20, 50, 100, 200]
SAMPLE_SEEDS = [i for i in range(10)] # Use ten different seeds for each sample size

# ##############################################################################
# 2. Data Loading and Processing Functions
# ##############################################################################

def parse_filename(filename, known_models):
    """Parses filename to get model name and multimodal status."""
    basename = os.path.basename(filename)
    found_model = next((model for model in known_models if model in basename), None)
    if not found_model:
        return None, None
    pre_model_part = basename.split(found_model)[0]
    multimodal = '_true_' in pre_model_part
    return found_model, multimodal

def load_and_prepare_data(files, known_models, split_map_file):
    """
    Loads data from .jsonl files, filters for 'release' split, and organizes
    scores by model and uuid.
    """
    try:
        with open(split_map_file, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        release_uuids = {uuid for uuid, data in split_map.items() if data.get('split') == 'release'}
        print(f"Loaded split map. Found {len(release_uuids)} questions in the 'release' split.")
    except FileNotFoundError:
        print(f"Error: '{split_map_file}' not found. Aborting.")
        return None, None

    # Structure: {model_name: {uuid: [scores]}}
    model_scores = defaultdict(lambda: defaultdict(list))
    for filepath in files:
        model_name, multimodal = parse_filename(filepath, known_models)
        if not model_name:
            continue
        
        # Store scores temporarily with multimodal flag
        key = (model_name, multimodal)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record['uuid'] in release_uuids:
                        # We only care about the first attempt (pass@1)
                        model_scores[key][record['uuid']].append(record['score'])
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # Consolidate data: Prefer Multimodal over Text-Only
    final_model_scores = defaultdict(dict)
    for (model_name, multimodal), scores_dict in model_scores.items():
        if multimodal: # Always prioritize multimodal results
            final_model_scores[model_name] = scores_dict
        elif model_name not in final_model_scores: # Only use text if no multi exists
            final_model_scores[model_name] = scores_dict

    # Ensure all models have scores for the same set of questions for fair comparison
    # Find the intersection of UUIDs across all models
    if not final_model_scores:
        print("No valid model data found.")
        return None, None
    
    common_uuids = release_uuids.copy()
    # common_uuids = set.intersection(*[set(scores.keys()) for scores in final_model_scores.values()])
    print(f"Found {len(common_uuids)} questions answered by all models.")

    # Filter each model's scores to only include common UUIDs
    for model in final_model_scores:
        final_model_scores[model] = {uuid: final_model_scores[model][uuid] for uuid in common_uuids}

    return dict(final_model_scores), sorted(list(common_uuids))

def calculate_pass_at_1(scores_dict, question_uuids, seed=42):
    """
    Calculates pass@1 accuracy for a given model's scores on a specific set of questions.
    """
        
    correct_count = 0
    for uuid in question_uuids:
        # Get scores for this uuid, could have multiple attempts
        attempts = scores_dict.get(uuid, [])
        if not attempts:
            continue
            
        # Use a consistent random choice for the single attempt
        rng = np.random.RandomState(seed)
        # Choose one result from the available attempts for this question
        chosen_score = rng.choice(attempts)
        if chosen_score == 1:
            correct_count += 1
            
    return (correct_count / len(question_uuids)) * 100

# ##############################################################################
# 3. Main Analysis Logic
# ##############################################################################

def main():
    """Main function to run the full analysis."""
    all_json_files = glob.glob('data/*.jsonl')
    if not all_json_files:
        print("No .jsonl files found in the current directory.")
        return

    print("--- Step 1: Loading and Preparing Data ---")
    model_data, all_release_uuids = load_and_prepare_data(all_json_files, KNOWN_MODELS, SPLIT_MAP_FILE)
    if not model_data:
        print("Data loading failed. Exiting.")
        return

    models_to_analyze = sorted(model_data.keys())
    print(f"\nAnalyzing the following {len(models_to_analyze)} models: {models_to_analyze}")
    
    print("\n--- Step 2: Calculating Benchmark Rank on Full Release Dataset ---")
    full_set_accuracies = {
        model: calculate_pass_at_1(model_data[model], all_release_uuids, seed=42)
        for model in models_to_analyze
    }
    
    # Create a DataFrame for the full set results and get the rank
    df_full = pd.DataFrame.from_dict(full_set_accuracies, orient='index', columns=['Accuracy'])
    df_full['Rank'] = df_full['Accuracy'].rank(method='min', ascending=False)
    df_full = df_full.sort_values('Rank')
    benchmark_ranks = df_full['Rank']
    
    print("Benchmark Ranks (based on pass@1, seed=42 on all release questions):")
    print(df_full)

    print("\n--- Step 3: Performing Sampling Experiments ---")
    results = []
    for n in SAMPLE_SIZES:
        if n > len(all_release_uuids):
            print(f"Sample size {n} is larger than the dataset size {len(all_release_uuids)}. Skipping.")
            continue
            
        for seed in SAMPLE_SEEDS:
            # Sample N questions from the release set
            rng = np.random.RandomState(seed)
            sampled_uuids = rng.choice(all_release_uuids, size=n, replace=False)
            
            # Calculate accuracies on this subset
            subset_accuracies = {
                model: calculate_pass_at_1(model_data[model], sampled_uuids, seed=42)
                for model in models_to_analyze
            }
            
            # Get the ranks for the subset
            df_subset = pd.DataFrame.from_dict(subset_accuracies, orient='index', columns=['Accuracy'])
            subset_ranks = df_subset['Accuracy'].rank(method='min', ascending=False)
            
            # Align ranks by model name before calculating correlation
            aligned_benchmark = benchmark_ranks.loc[models_to_analyze]
            aligned_subset = subset_ranks.loc[models_to_analyze]

            # Calculate Spearman correlation
            correlation, p_value = spearmanr(aligned_benchmark, aligned_subset)
            
            results.append({'SampleSize': n, 'Seed': seed, 'SpearmanCorrelation': correlation})
            print(f"N={n:3d}, Seed={seed}: Spearman's ρ = {correlation:.4f}")

    df_results = pd.DataFrame(results)

    print("\n--- Step 4: Plotting the Results ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use seaborn to plot the mean and confidence interval (or std dev)
    sns.lineplot(
        data=df_results,
        x='SampleSize',
        y='SpearmanCorrelation',
        marker='o',
        err_style="band", # 'band' shows confidence interval, 'bars' shows error bars
        ax=ax
    )

    ax.set_title('Stability of Model Ranking vs. Sample Size', fontsize=20, pad=15)
    ax.set_xlabel('Sample Size (N)', fontsize=14)
    ax.set_ylabel("Spearman's Rank Correlation (ρ)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='ρ = 0.95 (High Correlation)')
    
    # Make x-axis logarithmic to better visualize smaller N values
    ax.set_xscale('log')
    ax.set_xticks(SAMPLE_SIZES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Use normal numbers for ticks

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("results/ranking_stability_vs_sample_size.png", dpi=300)
    print("\nPlot saved to 'results/ranking_stability_vs_sample_size.png'")
    plt.show()

if __name__ == "__main__":
    main()