import os
import json
import glob
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors

# ##############################################################################
# 1. Configuration Area
# ##############################################################################
RESULTS_DIR = 'data/'  # Directory where .jsonl result files are stored

# [NEW] Path to the original questions file (required for extracting weights)
# Supports .parquet or .jsonl formats
QUESTIONS_FILE_PATH = 'data/20251014164938_questions.parquet'

SOTA_MODELS = [
    "gpt-5_high", "gemini-2_5-pro", "deepseek-v3_1-think-128k",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507"
]

KNOWN_MODELS = [
    "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3_1-think-128k", 
    "gemini-2_5-pro", "gpt-5_high", "qwen3-235b-a22b-instruct-2507", 
    # "gpt-5_medium", "gpt-5_low", "deepseek-v3_1-128k", "o3-2025-04-16", 
    # "gpt-4_1", "gpt-4o", "gpt-5_minimal"
]

PRETTY_NAMES_MAP = {
    "gpt-5_high": "GPT-5 (High)", "gpt-5_medium": "GPT-5 (Medium)",
    "gpt-5_low": "GPT-5 (Low)", "gemini-2_5-pro": "Gemini 2.5 Pro",
    "o3-2025-04-16": "o3 (High)", "deepseek-v3_1-think-128k": "DeepSeek-V3.1-Think",
    "deepseek-v3_1-128k": "DeepSeek-V3.1", "qwen3-235b-a22b-thinking-2507": "Qwen3-235B-Think",
    "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-Instruct",
    "gpt-4_1": "GPT-4.1", "gpt-4o": "GPT-4o", "gpt-5_minimal": "GPT-5 (Minimal)"
}

SPLIT_MAP_FILE = 'data/dataset_split_map.json'

# ##############################################################################
# 2. Helper Functions: Parse Weights and Filenames
# ##############################################################################

def extract_weights_from_explanation(text):
    """
    Extract Checkpoint weights from explanation text.
    Format example: <Checkpoint>[Content](1)</Checkpoint> or <Checkpoint>[Content](0.5)</Checkpoint>
    """
    if not text:
        return []
    # Regex logic:
    # <Checkpoint> 
    # \[.*?\]   -> Match content inside brackets (non-greedy)
    # \s*       -> Allow optional whitespace between brackets and parentheses
    # \(        -> Left parenthesis
    # (\d+(?:\.\d+)?) -> Capture group: integer or float (e.g. 1 or 0.5)
    # \)        -> Right parenthesis
    # \s*       -> Allow whitespace
    # </Checkpoint>
    pattern = r'<Checkpoint>\[.*?\]\s*\((\d+(?:\.\d+)?)\)\s*</Checkpoint>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Convert extracted strings to floats
    try:
        weights = [float(w) for w in matches]
    except ValueError:
        print(f"Warning: Failed to parse weights in text snippet: {text[:50]}...")
        weights = []
    return weights

def load_question_weights(filepath):
    """
    Load original questions file, build uuid -> weights_list map
    """
    print(f"Loading question weights from {filepath}...")
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.jsonl'):
        df = pd.read_json(filepath, lines=True)
    else:
        print("Error: QUESTIONS_FILE_PATH must be .parquet or .jsonl")
        return {}
    
    uuid_weights_map = {}
    
    # Assume explanation column is 'explanation_en'
    # If Chinese version, change to 'explanation_zh'
    col_name = 'explanation_en' 
    
    if col_name not in df.columns:
        print(f"Error: Column '{col_name}' not found in dataset. Available columns: {df.columns}")
        return {}

    for _, row in df.iterrows():
        uuid = row['uuid']
        explanation = row[col_name]
        weights = extract_weights_from_explanation(explanation)
        if weights:
            uuid_weights_map[uuid] = weights
            
    print(f"Loaded weights for {len(uuid_weights_map)} questions.")
    return uuid_weights_map

def parse_filename(filename, known_models):
    basename = os.path.basename(filename)
    if '_EVAL_BY_' in basename:
        search_part = basename.split('_EVAL_BY_')[0]
    else:
        search_part = basename
    sorted_known_models = sorted(known_models, key=len, reverse=True)
    found_model = next((model for model in sorted_known_models if model in search_part), None)
    if not found_model: return None, None, None
    try:
        pre_model_part = basename.split(found_model)[0]
        multimodal = '_true_' in pre_model_part
        pass_k_str = basename.rsplit('.', 1)[0].rsplit('_', 1)[-1]
        pass_k = int(pass_k_str) if pass_k_str.isdigit() else 1
        return found_model, multimodal, pass_k
    except (IndexError, ValueError):
        return found_model, '_true_' in basename, 1

# ##############################################################################
# 3. Core Calculation Logic (Accuracy unchanged, RPF modified to weighted)
# ##############################################################################

def process_data_files(files, known_models, split_map_file):
    try:
        with open(split_map_file, 'r', encoding='utf-8') as f:
            valid_uuids = {uuid for uuid, data in json.load(f).items() if data.get('split') in ['release', 'holdout']}
    except FileNotFoundError:
        return None
    raw_data = defaultdict(lambda: {'scores': defaultdict(list), 'pass_k': 1})
    for filepath in files:
        if '_EVAL_BY_' in filepath: continue
        model_name, multimodal, pass_k = parse_filename(filepath, known_models)
        if not model_name: continue
        key = (model_name, multimodal)
        raw_data[key]['pass_k'] = max(raw_data[key]['pass_k'], pass_k)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record['uuid'] in valid_uuids:
                        raw_data[key]['scores'][record['uuid']].append(record['score'])
                except (json.JSONDecodeError, KeyError): continue
    accuracies = []
    for (model_name, multimodal), data in raw_data.items():
        scores_dict = data['scores']
        if not scores_dict: continue
        input_type = 'Multimodal' if multimodal else 'Text-Only'
        total_questions = len(scores_dict)
        if model_name in SOTA_MODELS and data['pass_k'] >= 8:
            scores_matrix = np.array([scores[:8] for scores in scores_dict.values() if len(scores) >= 8])
            if scores_matrix.size > 0:
                acc_mean = np.mean(np.mean(scores_matrix, axis=0) * 100)
                accuracies.append({'Model': model_name, 'Input Type': input_type, 'Accuracy': acc_mean})
        else:
            first_trial_correct = sum(1 for scores in scores_dict.values() if scores and scores[0] == 1)
            acc_first_trial = (first_trial_correct / total_questions) * 100
            accuracies.append({'Model': model_name, 'Input Type': input_type, 'Accuracy': acc_first_trial})
    return pd.DataFrame(accuracies)

def calculate_rpf_weighted(files, known_models, split_map_file, uuid_weights_map):
    """
    Calculate weighted RPF score.
    Relies on both eval results and weights extracted from original questions.
    """
    try:
        with open(split_map_file, 'r', encoding='utf-8') as f:
            valid_uuids = {uuid for uuid, data in json.load(f).items() if data.get('split') in ['release', 'holdout']}
    except FileNotFoundError:
        return pd.DataFrame()
        
    rpf_data = defaultdict(list)
    
    for filepath in files:
        if '_EVAL_BY_' not in filepath: continue
        model_name, multimodal, _ = parse_filename(filepath, known_models)
        if not model_name: continue
        key = (model_name, 'Multimodal' if multimodal else 'Text-Only')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    uuid = record['uuid']
                    
                    if uuid not in valid_uuids:
                        continue
                        
                    # 1. Get weight list for the question
                    weights = uuid_weights_map.get(uuid, [])
                    if not weights:
                        # If weights not found (format error?), fallback to unweighted calculation
                        if record.get('total_checkpoints', 0) > 0:
                            rpf = (record.get('matched_checkpoints', 0) / record['total_checkpoints'])
                            rpf_data[key].append(rpf * 100)
                        continue

                    # 2. Get Eval details (True/False list)
                    # Note: eval_details must exist in your JSONL results
                    eval_details = record.get('evaluation_details', [])
                    if not eval_details:
                        # If no details, cannot map to weights, skip or fallback
                        continue
                        
                    # 3. Extract match status
                    # Assume LLM evaluates Checkpoints in order, matching weights order
                    matches = [item.get('is_matched', False) for item in eval_details]
                    
                    # 4. Length check
                    if len(weights) != len(matches):
                        # Mismatch usually due to LLM hallucination (extra or missing items)
                        # Conservative strategy: use min length of both
                        min_len = min(len(weights), len(matches))
                        weights = weights[:min_len]
                        matches = matches[:min_len]
                    
                    # 5. Calculate weighted score
                    # Numerator: sum(weight_i * 1 if matched else 0)
                    # Denominator: sum(weights)
                    total_weight = sum(weights)
                    if total_weight == 0:
                        rpf_score = 0
                    else:
                        earned_weight = sum(w for w, m in zip(weights, matches) if m)
                        # print(earned_weight, total_weight) # Debug print
                        rpf_score = (earned_weight / total_weight) * 100
                    
                    rpf_data[key].append(rpf_score)
                    
                except (json.JSONDecodeError, KeyError): 
                    continue
                    
    avg_rpf = []
    for (model_name, input_type), scores in rpf_data.items():
        if scores:
            avg_rpf.append({'Model': model_name, 'Input Type': input_type, 'RPF': np.mean(scores)})
            
    return pd.DataFrame(avg_rpf)

def merge_and_prioritize_data(accuracy_df, rpf_df):
    merged_df = pd.merge(accuracy_df, rpf_df, on=['Model', 'Input Type'], how='outer')
    final_records = []
    for model, group in merged_df.groupby('Model'):
        if len(group) == 1:
            final_records.append(group.iloc[0])
        else:
            multi_row = group[group['Input Type'] == 'Multimodal']
            if not multi_row.empty:
                final_records.append(multi_row.iloc[0])
            else:
                final_records.append(group[group['Input Type'] == 'Text-Only'].iloc[0])
    final_df = pd.DataFrame(final_records).dropna(subset=['Accuracy', 'RPF'])
    return final_df

# ##############################################################################
# 4. Plotting Function
# ##############################################################################
def plot_correlation_plotly_final(df):
    if df.empty:
        print("Final DataFrame is empty. Cannot plot."); return None
    df['Pretty Model'] = df['Model'].map(PRETTY_NAMES_MAP).fillna(df['Model'])
    fig = go.Figure()
    palette = pcolors.qualitative.Plotly
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'pentagon', 'hexagon', 'octagon']
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Accuracy']], y=[row['RPF']],
            mode='markers+text',
            text=[row['Pretty Model']],
            textposition='top center' if ('Low' not in row['Pretty Model'] and "GPT-5 (High)" not in row['Pretty Model']) else 'bottom center',
            textfont=dict(size=20, color='#000000', family="Arial", weight="bold"),
            marker=dict(color=palette[i % len(palette)], symbol=markers[i % len(markers)], size=24, line=dict(width=2, color='black')),
            hoverinfo='none', showlegend=False,
        ))
    avg_acc = df['Accuracy'].mean()
    avg_rpf = df['RPF'].mean()
    fig.add_vline(x=avg_acc, line_width=2, line_dash="dash", line_color="grey")
    fig.add_hline(y=avg_rpf, line_width=2, line_dash="dash", line_color="grey")
    
    quadrant_annotations = [
        dict(
            x=0.98, y=0.98, xref="paper", yref="paper", 
            text="<b>High Accuracy, High Fidelity</b><br>(Ideal)", 
            showarrow=False, xanchor='right', yanchor='top', 
            font=dict(color='darkgreen', size=22),
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="darkgreen", borderwidth=2, borderpad=10
        ),
        dict(
            x=0.02, y=0.98, xref="paper", yref="paper", 
            text="<b>Low Accuracy, High Fidelity</b><br>(Correct Process, Flawed Execution)", 
            showarrow=False, xanchor='left', yanchor='top', 
            font=dict(color='darkorange', size=22),
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="darkorange", borderwidth=2, borderpad=10
        ),
        dict(
            x=0.98, y=0.02, xref="paper", yref="paper", 
            text="<b>High Accuracy, Low Fidelity</b><br>(Heuristic / Shortcut-Driven)", 
            showarrow=False, xanchor='right', yanchor='bottom', 
            font=dict(color='red', size=22),
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="red", borderwidth=2, borderpad=10
        ),
        dict(
            x=0.02, y=0.02, xref="paper", yref="paper", 
            text="<b>Low Accuracy, Low Fidelity</b><br>(Needs Improvement)", 
            showarrow=False, xanchor='left', yanchor='bottom', 
            font=dict(color='dimgray', size=22),
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="dimgray", borderwidth=2, borderpad=10
        )
    ]
    for ann in quadrant_annotations: fig.add_annotation(ann)
    
    fig.update_layout(
        showlegend=False, 
        xaxis_title=dict(text='<b>Answer Accuracy (pass@1, %)</b>', font=dict(size=24, color='black')),
        yaxis_title=dict(text='<b>Reasoning Path Fidelity (RPF, %)</b>', font=dict(size=24, color='black')),
        font=dict(family="Arial, sans-serif", color="black"), 
        template='plotly_white', 
        height=900, width=1300,
        margin=dict(l=100, r=50, b=100, t=100),
        xaxis_range=[df['Accuracy'].min() - 5, df['Accuracy'].max() + 5],
        yaxis_range=[df['RPF'].min() - 6, df['RPF'].max() + 6],
        xaxis=dict(
            tickfont=dict(size=20, color='black'), 
            gridcolor='#e0e0e0', gridwidth=1,
            showline=True, linewidth=3, linecolor='black', mirror=True
        ), 
        yaxis=dict(
            tickfont=dict(size=20, color='black'), 
            gridcolor='#e0e0e0', gridwidth=1,
            showline=True, linewidth=3, linecolor='black', mirror=True
        ),
        plot_bgcolor='white'
    )
    return fig

# ##############################################################################
# 5. Main Execution
# ##############################################################################
if __name__ == "__main__":
    # Check if questions file exists
    if not os.path.exists(QUESTIONS_FILE_PATH):
        print(f"Error: Please set 'QUESTIONS_FILE_PATH' to your parquet/jsonl file. Current: {QUESTIONS_FILE_PATH}")
        exit()

    all_files = glob.glob(os.path.join(RESULTS_DIR, '*.jsonl'))
    if not all_files:
        print(f"No .jsonl files found in the directory: '{RESULTS_DIR}'"); exit()

    print("Step 0: Loading Question Weights...")
    uuid_weights_map = load_question_weights(QUESTIONS_FILE_PATH)
    if not uuid_weights_map:
        print("Failed to load weights. Exiting."); exit()

    print("\nStep 1: Calculating pass@1 accuracies...")
    accuracy_df = process_data_files(all_files, KNOWN_MODELS, SPLIT_MAP_FILE)
    
    print("\nStep 2: Calculating WEIGHTED Reasoning Path Fidelity (RPF)...")
    # Use new weighted function
    rpf_df = calculate_rpf_weighted(all_files, KNOWN_MODELS, SPLIT_MAP_FILE, uuid_weights_map)
    
    if accuracy_df is not None and not rpf_df.empty:
        print("\nStep 3: Merging data and prioritizing multimodal results...")
        final_df = merge_and_prioritize_data(accuracy_df, rpf_df)
        
        print("\n--- Final Data for Plotting ---")
        print(final_df.to_string())
        
        if not final_df.empty:
            print("\nStep 4: Generating final correlation plot with Plotly...")
            fig = plot_correlation_plotly_final(final_df)
            
            if fig:
                html_out = "results/accuracy_vs_rpf_weighted_final.html"
                pdf_out = "results/accuracy_vs_rpf_weighted_final.pdf"
                fig.write_html(html_out)
                try:
                    fig.write_image(pdf_out, format='pdf', width=1200, height=850, scale=2)
                    print(f"\n[output] Saved chart to '{html_out}' and '{pdf_out}'")
                except Exception as e:
                    print(f"\n[output] Saved chart to '{html_out}'")
                    print(f"Could not save PDF. Error: {e}")
        else:
            print("No models with complete data.")
    else:
        print("Could not generate plot.")
