import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import textwrap

# Plotly for modern, interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors

# ##############################################################################
# 1. Configuration Area (Keep as is)
# ##############################################################################

# Please set RESULTS_DIR to the directory where you store .jsonl files
RESULTS_DIR = 'data/' 

CHECKPOINT_TAGS_FILE = 'data/ckpt_20250915021133_questions_en_true_gemini-2_5-pro_high_1_0_1.jsonl'
ABILITY_TAG_DESC_FILE = 'data/ability_tags_description.json'
SPLIT_MAP_FILE = 'data/dataset_split_map.json'

KNOWN_MODELS = [
    "qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3_1-think-128k", "deepseek-v3_1-128k",
    "gemini-2_5-pro_high", "o3-2025-04-16_high",
    "gpt-5_high", "gpt-5_medium", "gpt-5_low",
    "gpt-4_1", "gpt-4o-2024-11-20", "gpt-5_minimal"
]

MODELS_TO_PLOT = [
    'gpt-5_high',
    'gemini-2_5-pro_high',
    'deepseek-v3_1-think-128k',
]

model_name_mapping = {
    'gpt-5_high': 'GPT-5 (High)',
    'gemini-2_5-pro_high': 'Gemini-2.5-Pro',
    'deepseek-v3_1-think-128k': 'DeepSeek-v3.1-Think',
}

# ##############################################################################
# 2. Data Loading and Processing Functions (Keep your logic completely)
# ##############################################################################
def parse_filename_for_eval(filename, known_models):
    basename = os.path.basename(filename)
    search_part = basename.split('_EVAL_BY_')[0]
    sorted_known_models = sorted(known_models, key=len, reverse=True)
    found_model = None
    for model in sorted_known_models:
        if model in search_part:
            found_model = model
            break
    if not found_model: return None, None
    pre_model_part = basename.split(found_model)[0]
    multimodal = '_true_' in pre_model_part
    return found_model, multimodal

def load_split_map(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            split_map = json.load(f)
        valid_uuids = {uuid for uuid, data in split_map.items() if data.get('split') in ['release', 'holdout']}
        print(f"Loaded split map. Found {len(valid_uuids)} non-easy questions.")
        return valid_uuids
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Will not filter easy questions.")
        return None

def load_checkpoint_tags(filepath):
    checkpoint_tags = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                uuid = record['uuid']
                tags_per_cp = {i: analysis.get('ability_tags', []) for i, analysis in enumerate(record.get('checkpoints_analysis', []))}
                checkpoint_tags[uuid] = tags_per_cp
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"Loaded ground truth checkpoint tags for {len(checkpoint_tags)} questions.")
    return checkpoint_tags

def load_rpf_evaluation_data(results_dir, known_models, valid_uuids):
    rpf_data = defaultdict(dict)
    files = glob.glob(os.path.join(results_dir, '*_EVAL_BY_*.jsonl'))
    print(f"Found {len(files)} RPF evaluation files to process...")
    for filepath in files:
        model_name, multimodal = parse_filename_for_eval(filepath, known_models)
        if not model_name: continue
        key = (model_name, multimodal)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    uuid = record.get('uuid')
                    if valid_uuids is None or uuid in valid_uuids:
                        if 'evaluation_details' in record and isinstance(record['evaluation_details'], list):
                            matched_statuses = [detail.get('is_matched', False) for detail in record['evaluation_details']]
                            rpf_data[key][uuid] = matched_statuses
                except (json.JSONDecodeError, KeyError):
                    continue
    print("Finished loading RPF evaluation data.")
    return rpf_data

def create_checkpoint_level_dataframe(rpf_data, checkpoint_tags):
    print("Creating checkpoint-level performance DataFrame...")
    records = []
    for (model, multimodal), evals_by_uuid in rpf_data.items():
        for uuid, matched_statuses in evals_by_uuid.items():
            if uuid not in checkpoint_tags: continue
            tags_for_uuid = checkpoint_tags[uuid]
            if len(matched_statuses) != len(tags_for_uuid): continue
            for cp_idx, is_matched in enumerate(matched_statuses):
                score = 1 if is_matched else 0
                tags = tags_for_uuid.get(cp_idx, [])
                for tag in tags:
                    records.append({'model': model, 'multimodal': multimodal, 'uuid': uuid, 'checkpoint_index': cp_idx, 'score': score, 'ability_tag': tag})
    if not records:
        print("Warning: No valid records were created. The resulting DataFrame will be empty.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    print(f"DataFrame created with {len(df)} checkpoint-level records.")
    df['main_category'] = df['ability_tag'].str.split('.').str[0]
    return df

def calculate_ability_scores(df):
    if df.empty: return None, None
    main_scores = df.groupby(['model', 'multimodal', 'main_category'])['score'].mean().unstack(level='main_category') * 100
    sub_scores = df.groupby(['model', 'multimodal', 'ability_tag'])['score'].mean() * 100
    return main_scores, sub_scores

def flatten_descriptions(desc_raw):
    flat = {}
    if isinstance(desc_raw, dict):
        flat.update(desc_raw.get('main_categories', {}))
        flat.update(desc_raw.get('sub_categories', {}))
    return flat

# ##############################################################################
# 3. (REBUILT) PLOTLY-BASED COMBINED VISUALIZATION FUNCTION
# ##############################################################################

def plot_combined_radar_plotly(sub_scores, models_to_plot, descriptions, palette):
    """
    (Corrected) Use Plotly to create a single chart containing 2x2 radar subplots with a shared legend.
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'polar'}],
               [{'type': 'polar'}, {'type': 'polar'}]],
        subplot_titles=[f"<b>{descriptions.get(str(i), f'Category {i}')}</b>" for i in range(1, 5)],
        vertical_spacing=0.2, # Increase vertical spacing
        horizontal_spacing=0.
    )

    for i, main_cat_id in enumerate(['1', '2', '3', '4']):
        row, col = i // 2 + 1, i % 2 + 1
        
        all_sub_tags_in_cat = sorted(
            [tag for tag in sub_scores.index.get_level_values('ability_tag').unique() if tag.startswith(f"{main_cat_id}.")],
            key=lambda t: tuple(map(float, t.split(' ')[:1]))
        )
        if not all_sub_tags_in_cat: continue

        # MODIFICATION 2: Wrap the angle axis labels (thetas)
        thetas = [textwrap.fill(descriptions.get(t, t), width=25).replace('\n', '<br>') for t in all_sub_tags_in_cat]

        for j, model in enumerate(models_to_plot):
            color = palette[j]
            
            if (model, True) in sub_scores.index: series = sub_scores.loc[(model, True)]
            elif (model, False) in sub_scores.index: series = sub_scores.loc[(model, False)]
            else: continue

            values = series.reindex(all_sub_tags_in_cat, fill_value=0).tolist()
            
            # MODIFICATION 1: Append the first data point to the end to close the curve
            values_closed = values + [values[0]]
            thetas_closed = thetas + [thetas[0]]
            
            # Correct the legend name, no longer wrap
            model_name_clean = model_name_mapping.get(model, model)

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=thetas_closed,
                fill='none',
                showlegend=(i == 0),
                name=model_name_clean,
                line=dict(color=color, width=6)
            ), row=row, col=col)

    # Define common polar style settings
    polar_style = dict(
        bgcolor='white',
        radialaxis=dict(
            range=[0, 101],
            tickvals=[25, 50, 75, 100],
            tickfont=dict(size=24, family="Arial Black", color="black"),
            gridcolor="#B0B0B0",
            gridwidth=3,
            layer='below traces'
        ),
        angularaxis=dict(
            tickfont=dict(size=30),
            gridcolor="#B0B0B0",
            gridwidth=3,
            layer='below traces',
            linecolor="#B0B0B0",
            linewidth=3,
            showline=True
        )
    )

    fig.update_layout(
        font_family="Arial",
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=2200,
        width=3000,
        margin=dict(t=180, b=250, l=0, r=0),
        legend=dict(
            orientation="h",
            # MODIFICATION 3: Further lower the legend position
            yanchor="bottom", y=-0.12,
            xanchor="center", x=0.5,
            font=dict(family="Arial", size=48, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=2,
            itemwidth=100
        ),
        # Unify the style of all subplots
        polar=polar_style,
        polar2=polar_style,
        polar3=polar_style,
        polar4=polar_style
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 42
        annotation.yshift = 110 # Move the title up to make room for long labels

    return fig

# ##############################################################################
# 4. Main Execution Area (Keep as is)
# ##############################################################################
if __name__ == "__main__":
    valid_uuids = load_split_map(SPLIT_MAP_FILE)
    checkpoint_tags_gt = load_checkpoint_tags(CHECKPOINT_TAGS_FILE)
    rpf_evaluations = load_rpf_evaluation_data(RESULTS_DIR, KNOWN_MODELS, valid_uuids)

    try:
        with open(ABILITY_TAG_DESC_FILE, 'r', encoding='utf-8') as f:
            raw_desc = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ability description file '{ABILITY_TAG_DESC_FILE}' not found. Aborting."); exit()

    ability_descriptions = flatten_descriptions(raw_desc)
    perf_df = create_checkpoint_level_dataframe(rpf_evaluations, checkpoint_tags_gt)
    main_category_scores, sub_category_scores = calculate_ability_scores(perf_df)
    
    if sub_category_scores is None or sub_category_scores.empty:
        print("No sub-category scores were generated. Cannot create radar charts.")
    else:
        print("\n--- Sub-category Scores ---")
        df_print = sub_category_scores.reset_index(name='score')
        unique_tags = df_print['ability_tag'].unique()
        try:
            unique_tags = sorted(unique_tags, key=lambda t: [int(x) if x.isdigit() else x for x in t.split(' ')[0].split('.')])
        except:
            unique_tags = sorted(unique_tags)

        for tag in unique_tags:
            desc = ability_descriptions.get(tag, tag)
            print(f"\n[Ability] {tag}: {desc}")
            tag_data = df_print[df_print['ability_tag'] == tag].sort_values(by=['model', 'multimodal'])
            for _, row in tag_data.iterrows():
                model_name = row['model']
                is_mm = row['multimodal']
                score_val = row['score']
                mode_str = "Multimodal" if is_mm else "Text-only"
                print(f"  - {model_name} ({mode_str}): {score_val:.2f}")

        print("\n--- Generating Combined Plotly Radar Chart ---")
        # Use a more vibrant color palette
        palette = pcolors.qualitative.T10
        
        fig = plot_combined_radar_plotly(sub_category_scores, MODELS_TO_PLOT, ability_descriptions, palette)

        if fig:
            html_out = "results/radar_combined_panel.html"
            png_out = "results/radar_combined_panel.png"
            svg_out = "results/radar_combined_panel.svg" # <-- New SVG filename
            pdf_out = "results/radar_combined_panel.pdf" # <-- New PDF filename
            try:
                # Save as high-resolution PNG
                fig.write_image(png_out, scale=3) # It is recommended to set scale to 3 or higher for a clearer PNG
                print(f"[output] Saved static image to {png_out}")

                # Save as SVG (vector image, recommended)
                fig.write_image(svg_out)
                print(f"[output] Saved vector image to {svg_out}")

                # Save as PDF (vector image)
                fig.write_image(pdf_out)
                print(f"[output] Saved vector image to {pdf_out}")

            except Exception as e:
                # The message can be more generic as kaleido handles all these formats
                print(f"Error saving static image. Make sure 'kaleido' is installed (`pip install kaleido`). Error: {e}")