#!/usr/bin/env python3
"""
Streamlit app for visualizing DAG evaluation results with Score vs RPF scatter plot.
Features:
- Model selection from cleaned directory
- Interactive Score vs RPF scatter plot (Plotly)
- DAG visualization with color coding for ground truth and LLM answers
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


# ============================================================
# Data Loading
# ============================================================

def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file"""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_data_file(data_dir: str, model_name: str, multimodal: bool) -> Optional[str]:
    """
    Find the corresponding LLM output data file in the data directory.
    File pattern: 20251014164938_questions_release_en_{true/false}__{model_name}__*.jsonl
    """
    if not os.path.isdir(data_dir):
        return None
    
    multi_str = "true" if multimodal else "false"
    # Look for files matching the model name and multimodal flag
    for entry in os.listdir(data_dir):
        if not entry.endswith(".jsonl"):
            continue
        if "backup" in entry:
            continue
        # Match pattern: ...en_{true/false}__{model_name}__...
        if f"_en_{multi_str}__{model_name}__" in entry or f"_en_{multi_str}__{model_name}." in entry:
            return os.path.join(data_dir, entry)
    return None


def list_model_files(cleaned_dir: str) -> List[Dict[str, Any]]:
    """
    List all model files in cleaned directory and parse metadata.
    File pattern: match_results_{multimodal}__{model_name}__v5_merged.jsonl
    """
    if not os.path.isdir(cleaned_dir):
        return []
    
    models = []
    for entry in os.listdir(cleaned_dir):
        if not entry.endswith(".jsonl") or entry.startswith("ground_truth"):
            continue
        if "backup" in entry:
            continue
        
        # Parse filename: match_results_false__gpt-5_high__v5_merged.jsonl
        # Pattern: match_results_{true/false}__{model_name}__v5_merged...
        match = re.match(r"match_results_(true|false)__(.+?)__v5_merged", entry)
        if match:
            multimodal = match.group(1) == "true"
            model_name = match.group(2)
            models.append({
                "filename": entry,
                "model_name": model_name,
                "multimodal": multimodal,
                "display_name": f"{model_name} ({'Multimodal' if multimodal else 'Text-only'})",
                "path": os.path.join(cleaned_dir, entry)
            })
    
    return sorted(models, key=lambda x: x["model_name"].lower())


# ============================================================
# RPF Calculation (from notebook)
# ============================================================

def calculate_dag_similarity(gt_graph: dict, llm_graph: dict) -> dict:
    """
    Calculate structural similarity between LLM reasoning path and ground truth path
    
    Args:
        gt_graph: Ground truth graph {'nodes': [{'id', 'points'}], 'edges': [{'from', 'to'}]}
        llm_graph: LLM graph {'nodes': [{'id'}], 'edges': [{'from', 'to'}], 'matches': [{'h_id', 'r_id'}]}
    
    Returns:
        {
            'rpf': float,
            'node_recall': float, 'node_precision': float, 'node_f1': float,
            'edge_recall': float, 'edge_precision': float, 'edge_f1': float,
            'node_details': dict
        }
    """
    
    # 1. Build graph
    G_gt = nx.DiGraph()
    G_gt.add_nodes_from(n['id'] for n in gt_graph.get('nodes', []))
    G_gt.add_edges_from((e['from'], e['to']) for e in gt_graph.get('edges', []))
    gt_closure = nx.transitive_closure(G_gt) if G_gt.nodes() else nx.DiGraph()
    
    G_llm = nx.DiGraph()
    G_llm.add_nodes_from(n['id'] for n in llm_graph.get('nodes', []))
    G_llm.add_edges_from((e['from'], e['to']) for e in llm_graph.get('edges', []))
    llm_closure = nx.transitive_closure(G_llm) if G_llm.nodes() else nx.DiGraph()
    
    # 2. Build mapping (supporting many-to-one)
    gt_to_llm = defaultdict(list)
    llm_to_gt = {}
    for m in llm_graph.get('matches', []):
        if m.get('r_id') is None or m.get('h_id') is None:
            continue
        gt_to_llm[m['r_id']].append(m['h_id'])
        llm_to_gt[m['h_id']] = m['r_id']
    
    # 3. Calculate node metrics
    gt_nodes = {n['id']: n for n in gt_graph.get('nodes', [])}
    gt_points = {n['id']: n.get('points', 1) for n in gt_graph.get('nodes', [])}
    total_score = 0.0
    max_score = sum(gt_points.values())
    node_details = {}
    
    for r_id, points in gt_points.items():
        h_ids = gt_to_llm.get(r_id, [])
        
        # Get complete node information
        gt_node = gt_nodes[r_id].copy()
        
        if not h_ids:
            node_details[r_id] = {**gt_node, 'matched': False, 'logic_ratio': 0, 'score': 0}
            continue
        
        parents = list(G_gt.predecessors(r_id))
        if not parents:
            logic_ratio = 1.0
        else:
            valid = sum(
                1 for p_id in parents
                if any(llm_closure.has_edge(p_h, h) 
                       for p_h in gt_to_llm.get(p_id, []) 
                       for h in h_ids)
            )
            logic_ratio = valid / len(parents)
        
        score = points * logic_ratio
        total_score += score
        node_details[r_id] = {**gt_node, 'matched': True, 'logic_ratio': logic_ratio, 'score': score}
    
    rpf = total_score / max_score if max_score > 0 else 0
    
    # Node recall / precision / f1
    matched_gt_nodes = set(gt_to_llm.keys()) & set(G_gt.nodes())
    matched_llm_nodes = set(llm_to_gt.keys()) & set(G_llm.nodes())
    
    node_recall = len(matched_gt_nodes) / len(G_gt.nodes()) if G_gt.nodes() else 0
    node_precision = len(matched_llm_nodes) / len(G_llm.nodes()) if G_llm.nodes() else 0
    node_f1 = 2 * node_recall * node_precision / (node_recall + node_precision) if (node_recall + node_precision) > 0 else 0
    
    # 4. Calculate edge metrics
    edge_results = {}
    
    # Edge recall: ground truth edges preserved in LLM transitive closure
    gt_edge_hits = 0
    for e in gt_graph.get('edges', []):
        from_llms = gt_to_llm.get(e['from'], [])
        to_llms = gt_to_llm.get(e['to'], [])
        hit = any(llm_closure.has_edge(f, t) for f in from_llms for t in to_llms)
        edge_results[(e['from'], e['to'])] = hit
        if hit:
            gt_edge_hits += 1
    
    edge_recall = gt_edge_hits / len(gt_graph.get('edges', [])) if gt_graph.get('edges') else 0
    
    # Edge precision: LLM edges preserved in ground truth transitive closure
    llm_edges = llm_graph.get('edges', [])
    llm_edge_hits = 0
    for e in llm_edges:
        from_gt = llm_to_gt.get(e['from'])
        to_gt = llm_to_gt.get(e['to'])
        if from_gt and to_gt and gt_closure.has_edge(from_gt, to_gt):
            llm_edge_hits += 1
    
    edge_precision = llm_edge_hits / len(llm_edges) if llm_edges else 0
    edge_f1 = 2 * edge_recall * edge_precision / (edge_recall + edge_precision) if (edge_recall + edge_precision) > 0 else 0
    
    return {
        'rpf': rpf,
        'node_recall': node_recall,
        'node_precision': node_precision,
        'node_f1': node_f1,
        'edge_recall': edge_recall,
        'edge_precision': edge_precision,
        'edge_f1': edge_f1,
        'node_details': node_details,
        'edge_results': edge_results,
        'gt_to_llm': dict(gt_to_llm),
        'llm_to_gt': llm_to_gt,
        'total_score': total_score,
        'max_score': max_score
    }


def calculate_score_for_llm(gt_graph: dict, llm_graph: dict) -> float:
    """Calculate DAG Score for LLM Answer (normalized percentage)"""
    result = calculate_dag_similarity(gt_graph, llm_graph)
    return result['rpf'] * 100  # convert to percentage


# ============================================================
# Graph Rendering
# ============================================================

def render_graph(graph_data: dict, results: dict, is_ref: bool = True, graph_key: str = "graph"):
    """Render DAG using streamlit_agraph"""
    nodes_list = []
    edges_list = []
    
    for n in graph_data.get('nodes', []):
        node_id = n['id']
        
        if is_ref:
            # Ground truth graph: color by match status
            r = results.get('node_details', {}).get(node_id, {})
            is_critical = n.get('critical', True)
            if not is_critical:
                color = '#CCCCCC'  # Gray: non-critical nodes
            elif r.get('logic_ratio', 0) == 1:
                color = '#90EE90'  # Light green: full match
            elif r.get('matched'):
                color = '#FFD700'  # Yellow: matched but logic incomplete
            else:
                color = '#FF6B6B'  # Red: not matched
            
            # Show only node ID and score
            points = n.get('points', 1)
            label = f"{node_id}\n{points}pts"
            # Hover tooltip shows full node content
            title = f"[{node_id}] {n.get('content', 'No content')[:200]}"
        else:
            # LLM answer graph: color by whether matched
            matched_r_id = results.get('llm_to_gt', {}).get(node_id)
            if matched_r_id:
                color = '#90EE90'  # Green: matched
            else:
                color = '#FFB6C1'  # Pink: not matched (possibly hallucination)
            
            # Show only node ID and match relationship
            label = node_id
            if matched_r_id:
                label += f"\n→ {matched_r_id}"
            # Hover tooltip shows full node content
            title = f"[{node_id}] {n.get('content', 'No content')[:200]}"
        
        # Add prefix to node IDs to ensure uniqueness
        unique_node_id = f"{graph_key}_{node_id}"
        nodes_list.append(Node(
            id=unique_node_id,
            label=label,
            size=25,
            color=color,
            font={'size': 12},
            title=title
        ))
    
    for e in graph_data.get('edges', []):
        from_id = e['from']
        to_id = e['to']
        
        if is_ref:
            matched = results.get('edge_results', {}).get((from_id, to_id), False)
            color = '#90EE90' if matched else '#FF6B6B'
        else:
            color = '#999999'
        
        # Add same prefix to edge node IDs
        edges_list.append(Edge(
            source=f"{graph_key}_{from_id}",
            target=f"{graph_key}_{to_id}",
            color=color
        ))
    
    config = Config(
        width='100%',
        height=500,
        directed=True,
        physics={
            'enabled': True,
            'hierarchicalRepulsion': {
                'centralGravity': 0.0,
                'springLength': 100,
                'springConstant': 0.01,
                'nodeDistance': 120,
                'damping': 0.09
            },
            'minVelocity': 0.75,
            'solver': 'hierarchicalRepulsion'
        },
        layout={'hierarchical': {'enabled': True, 'direction': 'UD', 'sortMethod': 'directed'}},
        hierarchical=True,
        interaction={
            'navigationButtons': False,
            'keyboard': False,
            'hover': True,
            'selectable': True,
            'dragNodes': True,
            'dragView': True,
            'zoomView': True
        }
    )
    
    if nodes_list:
        agraph(nodes=nodes_list, edges=edges_list, config=config)
    else:
        st.info("No nodes to display")


# ============================================================
# Main App
# ============================================================

def main():
    st.set_page_config(page_title="DAG Evaluation Viewer", layout="wide")
    st.title("📊 DAG Evaluation Visualization")
    
    # Sidebar: Model selection
    with st.sidebar:
        st.header("📂 Modelselect")
        
        base_dir = st.text_input(
            "Cleaned Directory",
            value="/hdd01/zxhuang/SUPERChem_eval/DAG_eval/cleaned",
            help="containing model evaluation results"
        )
        
        gt_path = st.text_input(
            "Ground Truth File",
            value="/hdd01/zxhuang/SUPERChem_eval/DAG_eval/cleaned/ground_truth_graphs_cleaned.jsonl",
            help="Ground Truth DAG file path"
        )
        
        models = list_model_files(base_dir)
        if not models:
            st.error("Model file not found")
            st.stop()
        
        model_options = {m["display_name"]: m for m in models}
        sorted_names = [m["display_name"] for m in models]
        # Default to gemini-2_5-pro_high (Multimodal)
        default_name = "gemini-2_5-pro_high (Multimodal)"
        default_idx = sorted_names.index(default_name) if default_name in sorted_names else 0
        selected_model_name = st.selectbox(
            "selectModel",
            options=sorted_names,
            index=default_idx,
            help="select model evaluation results to view"
        )
        
        selected_model = model_options[selected_model_name]
        
        st.divider()
        st.info(f"**Model**: {selected_model['model_name']}\n\n"
                f"**Multimodal**: {'Yes' if selected_model['multimodal'] else 'No'}")
    
    # Load data
    try:
        # Load ground truth
        gt_data = load_jsonl(Path(gt_path))
        gt_dict = {item["uuid"]: item for item in gt_data}
        
        # Load model results
        model_data = load_jsonl(Path(selected_model["path"]))
        model_dict = {item["uuid"]: item for item in model_data}
        
        # Load LLM output data from DAG_eval/data/
        data_dir = os.path.join(os.path.dirname(base_dir), "data")
        data_file = find_data_file(data_dir, selected_model["model_name"], selected_model["multimodal"])
        llm_output_dict = {}
        if data_file:
            llm_output_data = load_jsonl(Path(data_file))
            llm_output_dict = {item["uuid"]: item for item in llm_output_data}
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Calculate metrics for all questions
    results_list = []
    for uuid, llm_item in model_dict.items():
        gt_item = gt_dict.get(uuid)
        if not gt_item:
            continue
        
        # Get ground truth graph
        gt_graph = gt_item  # cleaned file has nodes/edges directly
        
        # Get LLM graph (llm_item has nodes/edges/matches)
        llm_graph = llm_item
        
        # Calculate similarity
        sim_result = calculate_dag_similarity(gt_graph, llm_graph)
        
        # Get score from file or calculate
        score = llm_item.get('score', 0)
        if isinstance(score, dict):
            score = score.get('pct', 0)
        
        results_list.append({
            'uuid': uuid,
            'score': score,
            'multimodal': llm_item.get('multimodal', False),
            'rpf': sim_result['rpf'] * 100,  # Convert to percentage
            'node_recall': sim_result['node_recall'] * 100,
            'node_precision': sim_result['node_precision'] * 100,
            'node_f1': sim_result['node_f1'] * 100,
            'edge_recall': sim_result['edge_recall'] * 100,
            'edge_precision': sim_result['edge_precision'] * 100,
            'edge_f1': sim_result['edge_f1'] * 100,
            'max_score': sim_result['max_score'],
            'total_score': sim_result['total_score']
        })
    
    if not results_list:
        st.error("No matching evaluation data found")
        st.stop()
    
    df = pd.DataFrame(results_list)
    
    st.sidebar.success(f"✓ Loaded {len(df)} questions")
    
    # ----------------------------------------------------
    # filters
    # ----------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("🔍 Filter Criteria")
    
    # RPF Range filter
    rpf_range = st.sidebar.slider(
        "RPF range (%)",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
        help="Filter questions in specific RPF range"
    )
    
    # Score filter
    score_filter = st.sidebar.multiselect(
        "Answer Result",
        options=['Correct (1)', 'Incorrect (0)'],
        default=['Correct (1)', 'Incorrect (0)'],
        help="Filter correct or incorrect questions"
    )
    
    # Multimodal filter
    multimodal_filter = st.sidebar.multiselect(
        "questionstype",
        options=['Multimodal (True)', 'Text-only (False)'],
        default=['Multimodal (True)', 'Text-only (False)'],
        help="Filter Multimodal or Text-only questions"
    )
    
    # Apply filters
    # 1. RPF filter
    df_filtered = df[
        (df['rpf'] >= rpf_range[0]) & 
        (df['rpf'] <= rpf_range[1])
    ]
    
    # 2. Score filter
    selected_scores = []
    if 'Correct (1)' in score_filter:
        selected_scores.append(1)
    if 'Incorrect (0)' in score_filter:
        selected_scores.append(0)
    
    df_filtered = df_filtered[df_filtered['score'].isin(selected_scores)]
    
    # 3. Multimodal filter
    selected_multimodal = []
    if 'Multimodal (True)' in multimodal_filter:
        selected_multimodal.append(True)
    if 'Text-only (False)' in multimodal_filter:
        selected_multimodal.append(False)
    
    df_filtered = df_filtered[df_filtered['multimodal'].isin(selected_multimodal)]
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No questions match filter criteria. Please adjust filters.")
        st.stop()
        
    st.sidebar.info(f"After filter: {len(df_filtered)} questions")
    
    # Use filtered dataframe for subsequent operations
    df = df_filtered
    
    # ----------------------------------------------------
    # Model Overall Statistics (in sidebar)
    # ----------------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("📈 Modeloverall metrics")
    
    # Calculate overall averages
    accuracy = df['score'].mean() * 100
    avg_rpf = df['rpf'].mean()
    avg_node_recall = df['node_recall'].mean()
    avg_node_precision = df['node_precision'].mean()
    avg_node_f1 = df['node_f1'].mean()
    avg_edge_recall = df['edge_recall'].mean()
    avg_edge_precision = df['edge_precision'].mean()
    avg_edge_f1 = df['edge_f1'].mean()
    
    st.sidebar.metric("Accuracy (Accuracy)", f"{accuracy:.1f}%")
    st.sidebar.metric("average RPF", f"{avg_rpf:.1f}%")
    
    # Node metrics in sidebar
    st.sidebar.markdown("**Node metrics**")
    st.sidebar.markdown(f"- Recall: **{avg_node_recall:.1f}%**")
    st.sidebar.markdown(f"- Precision: **{avg_node_precision:.1f}%**")
    st.sidebar.markdown(f"- F1: **{avg_node_f1:.1f}%**")
    
    # Edge metrics in sidebar
    st.sidebar.markdown("**Edge metrics**")
    st.sidebar.markdown(f"- Recall: **{avg_edge_recall:.1f}%**")
    st.sidebar.markdown(f"- Precision: **{avg_edge_precision:.1f}%**")
    st.sidebar.markdown(f"- F1: **{avg_edge_f1:.1f}%**")
    
    # Add score label for display
    df['score_label'] = df['score'].apply(lambda x: 'Correct (1)' if x == 1 else 'Incorrect (0)')
    
    # Initialize session state for selected uuid
    if 'selected_uuid' not in st.session_state:
        st.session_state.selected_uuid = df['uuid'].iloc[0] if len(df) > 0 else None
    
    # Create uuid to score mapping for quick lookup
    uuid_to_score = dict(zip(df['uuid'], df['score']))
    
    # Main content - two columns (left smaller for chart, right larger for DAG)
    col_scatter, col_detail = st.columns([2, 3])
    
    with col_scatter:
        st.subheader("📈 RPF Distribution (by answer Correctness)")
        
        # Create strip plot with jitter for binary score
        # Add small jitter to y-axis to separate overlapping points
        np.random.seed(42)
        df['score_jitter'] = df['score'] + np.random.uniform(-0.15, 0.15, len(df))
        
        # Create figure with subplots: box plot + strip plot
        fig = go.Figure()
        
        # Add box plots for each score category
        for score_val, color, name in [(0, '#FF6B6B', 'Incorrect (0)'), (1, '#90EE90', 'Correct (1)')]:
            subset = df[df['score'] == score_val]
            fig.add_trace(go.Box(
                x=subset['rpf'],
                y=[name] * len(subset),
                name=name,
                orientation='h',
                marker_color=color,
                boxpoints=False,
                line=dict(width=2),
                fillcolor=color,
                opacity=0.3,
                showlegend=False
            ))
        
        # Add scatter points with jitter
        for score_val, color, name in [(0, '#FF6B6B', 'Incorrect (0)'), (1, '#90EE90', 'Correct (1)')]:
            subset = df[df['score'] == score_val].copy()
            # Add vertical jitter within the box area
            subset['y_jitter'] = name
            
            fig.add_trace(go.Scatter(
                x=subset['rpf'],
                y=[name] * len(subset),
                mode='markers',
                name=name,
                marker=dict(
                    size=10,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                customdata=subset[['uuid', 'node_recall', 'node_precision']].values,
                hovertemplate="<b>UUID</b>: %{customdata[0]}<br>" +
                              "<b>RPF</b>: %{x:.1f}%<br>" +
                              "<b>Node Recall</b>: %{customdata[1]:.1f}%<br>" +
                              "<b>Node Precision</b>: %{customdata[2]:.1f}%<extra></extra>",
                showlegend=True
            ))
        
        fig.update_layout(
            height=350,
            xaxis_title="RPF (%)",
            yaxis_title="Answer Result",
            hovermode='closest',
            title=f"RPF Distribution - {selected_model['model_name']}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(categoryorder='array', categoryarray=['Incorrect (0)', 'Correct (1)'])
        )
        
        # Display chart with click selection
        event = st.plotly_chart(
            fig, 
            use_containinger_width=True, 
            key="scatter_plot",
            on_select="rerun",
            selection_mode="points"
        )
        
        # Handle chart selection
        if event and event.get("selection") and event["selection"].get("points"):
            point = event["selection"]["points"][0]
            # customdata is [uuid, node_recall, node_precision]
            if "customdata" in point:
                clicked_uuid = point["customdata"][0]
                # Only rerun if changed
                if clicked_uuid != st.session_state.get("selected_uuid"):
                    st.session_state.selected_uuid = clicked_uuid
                    # Update input widget state to reflect selection
                    st.session_state.uuid_input_widget = clicked_uuid
                    st.rerun()
        
        st.caption("💡 **Tip**: Click scatter to jump, or hover to copy UUID and paste below")
        
        # Statistics
        with st.expander("📊 Statistics", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                acc = df['score'].mean() * 100
                st.metric("Accuracy", f"{acc:.1f}%")
            with col2:
                st.metric("average RPF", f"{df['rpf'].mean():.1f}%")
            with col3:
                # RPF difference between correct and incorrect
                rpf_correct = df[df['score'] == 1]['rpf'].mean() if len(df[df['score'] == 1]) > 0 else 0
                rpf_incorrect = df[df['score'] == 0]['rpf'].mean() if len(df[df['score'] == 0]) > 0 else 0
                st.metric("Correct questions average RPF", f"{rpf_correct:.1f}%", 
                         delta=f"+{rpf_correct - rpf_incorrect:.1f}%" if rpf_correct > rpf_incorrect else f"{rpf_correct - rpf_incorrect:.1f}%")
            
            # Additional row for more stats
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Incorrect questions average RPF", f"{rpf_incorrect:.1f}%")
            with col5:
                st.metric("Correct questions count", f"{len(df[df['score'] == 1])}")
            with col6:
                st.metric("Incorrect questions count", f"{len(df[df['score'] == 0])}")
        
        # questions selection
        st.divider()
        
        # UUID search/select with text input
        all_uuids = df['uuid'].tolist()
        uuid_set = set(all_uuids)
        
        # Ensure selected_uuid is valid in current filtered list
        if st.session_state.selected_uuid not in uuid_set:
            st.session_state.selected_uuid = all_uuids[0] if all_uuids else None
        
        # Get current index for navigation
        current_idx = all_uuids.index(st.session_state.selected_uuid) if st.session_state.selected_uuid in all_uuids else 0
        
        # Navigation buttons BEFORE text input to handle state updates
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("◀ Previous", disabled=(current_idx == 0)):
                st.session_state.selected_uuid = all_uuids[current_idx - 1]
                st.rerun()
        
        with col_info:
            st.markdown(f"<center>**{current_idx + 1}** / {len(all_uuids)}</center>", unsafe_allow_html=True)
        
        with col_next:
            if st.button("Next ▶", disabled=(current_idx >= len(all_uuids) - 1)):
                st.session_state.selected_uuid = all_uuids[current_idx + 1]
                st.rerun()
        
        # Text input for UUID jump - use a separate key and sync manually
        uuid_input = st.text_input(
            "🔍 Input/Paste UUID Jump",
            value=st.session_state.selected_uuid or "",
            placeholder="Paste UUID and press Enter to jump...",
            key=f"uuid_input_{st.session_state.selected_uuid}"  # Dynamic key to force refresh
        )
        
        # Validate and update selection (only if input differs from current selection)
        if uuid_input and uuid_input.strip() != st.session_state.selected_uuid:
            uuid_input = uuid_input.strip()
            if uuid_input in uuid_set:
                st.session_state.selected_uuid = uuid_input
                st.rerun()
            else:
                # Try partial match
                matches = [u for u in all_uuids if uuid_input.lower() in u.lower()]
                if len(matches) == 1:
                    st.session_state.selected_uuid = matches[0]
                    st.rerun()
                elif len(matches) > 1:
                    st.warning(f"found {len(matches)} matches found, please enter a more complete UUID")
                else:
                    st.error("Matched UUID not found")
    
    with col_detail:
        st.subheader("🎯 DAG Visualization")
        
        # Use session state for selected uuid
        current_uuid = st.session_state.selected_uuid
        
        if current_uuid:
            # Get data for selected question
            gt_item = gt_dict.get(current_uuid, {})
            llm_item = model_dict.get(current_uuid, {})
            
            if gt_item and llm_item:
                # Calculate similarity for this question
                sim_result = calculate_dag_similarity(gt_item, llm_item)
                
                # Get score for this question
                current_score = uuid_to_score.get(current_uuid, 0)
                is_correct = current_score == 1
                
                # Display answer correctness prominently with link to question
                question_url = f"https://superchem.pku.edu.cn/questions/{current_uuid}"
                is_multimodal = llm_item.get('multimodal', False)
                mm_tag = "🖼️ Multimodal" if is_multimodal else "📝 Text-only"
                if is_correct:
                    st.success(f"✅ **Model Answer Correct** | {mm_tag} | UUID: `{current_uuid}` | [🔗 viewquestions]({question_url})")
                else:
                    st.error(f"❌ **Model Answer Wrong** | {mm_tag} | UUID: `{current_uuid}` | [🔗 viewquestions]({question_url})")
                
                # Display metrics - Row 2: Node metrics
                st.markdown("##### 📊 Current Question Metrics")
                col_rpf, col_dag_score, col_n1, col_n2, col_n3, col_e1, col_e2, col_e3 = st.columns(8)
                with col_rpf:
                    st.metric("RPF", f"{sim_result['rpf']*100:.1f}%")
                with col_dag_score:
                    st.metric("DAG Score", f"{sim_result['total_score']:.1f}/{sim_result['max_score']}")
                with col_n1:
                    st.metric("Node Recall", f"{sim_result['node_recall']*100:.1f}%")
                with col_n2:
                    st.metric("Node Precision", f"{sim_result['node_precision']*100:.1f}%")
                with col_n3:
                    st.metric("Node F1", f"{sim_result['node_f1']*100:.1f}%")
                with col_e1:
                    st.metric("Edge Recall", f"{sim_result['edge_recall']*100:.1f}%")
                with col_e2:
                    st.metric("Edge Precision", f"{sim_result['edge_precision']*100:.1f}%")
                with col_e3:
                    st.metric("Edge F1", f"{sim_result['edge_f1']*100:.1f}%")
                
                # DAG visualizations
                tab1, tab2 = st.tabs(["📘 Ground Truth DAG", "📝 LLM DAG"])
                
                with tab1:
                    st.markdown("**Legend**: 🟢 Full Match | 🟡 Partial Match | 🔴 Not Matched | ⚪ Non-critical Nodes")
                    with st.containinger(border=True):
                        render_graph(
                            gt_item, 
                            sim_result, 
                            is_ref=True, 
                            graph_key=f"gt_{current_uuid}"
                        )
                
                with tab2:
                    st.markdown("**Legend**: 🟢 Matched to GT Nodes | 🩷 Not Matched（may be hallucination）")
                    with st.containinger(border=True):
                        render_graph(
                            llm_item, 
                            sim_result, 
                            is_ref=False, 
                            graph_key=f"llm_{current_uuid}"
                        )
                
                # Node details
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    with st.expander("📘 Ground Truth Node Details", expanded=False):
                        node_details = sim_result.get('node_details', {})
                        # Sort by ID numerically (extract number from R1, R2, etc.)
                        def extract_number(node_id):
                            import re
                            match = re.search(r'\d+', node_id)
                            return int(match.group()) if match else 0
                        
                        for r_id in sorted(node_details.keys(), key=extract_number):
                            details = node_details[r_id]
                            matched = details.get('matched', False)
                            logic_ratio = details.get('logic_ratio', 0)
                            points = details.get('points', 1)
                            score = details.get('score', 0)
                            content = details.get('content', '')
                            
                            if not matched:
                                st.error(f"❌ **{r_id}**: 0/{points} pts\n\n{content}")
                            elif logic_ratio == 1:
                                st.success(f"✅ **{r_id}**: {score:.1f}/{points} pts\n\n{content}")
                            else:
                                st.warning(f"⚠️ **{r_id}**: {score:.1f}/{points} pts (logic ratio: {logic_ratio:.1%})\n\n{content}")
                
                with col_d2:
                    with st.expander("📝 LLM Node Details (H Nodes)", expanded=False):
                        llm_nodes = llm_item.get('nodes', [])
                        llm_to_gt = sim_result.get('llm_to_gt', {})
                        
                        # Sort by ID numerically (extract number from H1, H2, etc.)
                        def extract_number_h(node_id):
                            import re
                            match = re.search(r'\d+', node_id)
                            return int(match.group()) if match else 0
                        
                        llm_nodes.sort(key=lambda x: extract_number_h(x.get('id', '')))
                        
                        for node in llm_nodes:
                            h_id = node.get('id')
                            content = node.get('content', '')
                            matched_gt = llm_to_gt.get(h_id)
                            
                            if matched_gt:
                                st.success(f"🔗 **{h_id}** → {matched_gt}\n\n{content}")
                            else:
                                st.error(f"⚪ **{h_id}** (Not Matched/Hallucination)\n\n{content}")
                # LLM Output
                llm_output_item = llm_output_dict.get(current_uuid, {})
                if llm_output_item:
                    with st.expander("💬 LLM Raw Output", expanded=False):
                        col_meta1, col_meta2, col_meta3 = st.columns(3)
                        with col_meta1:
                            st.markdown(f"**Model**: `{llm_output_item.get('model', 'N/A')}`")
                        with col_meta2:
                            st.markdown(f"**LLM Answer**: `{llm_output_item.get('llm_answer', 'N/A')}`")
                        with col_meta3:
                            st.markdown(f"**Finish Reason**: `{llm_output_item.get('finish_reason', 'N/A')}`")
                        
                        # Show reasoning if available
                        llm_reasoning = llm_output_item.get('llm_reasoning', '')
                        if llm_reasoning:
                            st.markdown("---")
                            st.markdown("**🧠 Reasoning (Chain of Thought)**")
                            st.markdown(llm_reasoning)
                        
                        # Show output
                        llm_output_text = llm_output_item.get('llm_output', '')
                        if llm_output_text:
                            st.markdown("---")
                            st.markdown("**📝 Output (Model Output)**")
                            st.markdown(llm_output_text)
                else:
                    st.caption("⚠️ No LLM Raw Output data found for this question")
                
                # DAG JSON data
                col_json1, col_json2 = st.columns(2)
                with col_json1:
                    with st.expander("📄 Ground Truth DAG (JSON)", expanded=False):
                        st.json(gt_item)
                with col_json2:
                    with st.expander("📄 LLM DAG (JSON)", expanded=False):
                        st.json(llm_item)
            else:
                st.warning("No data found for this question")
        else:
            st.info("👆 Click points in scatter plot or select questions from dropdown to view details")


if __name__ == "__main__":
    main()
