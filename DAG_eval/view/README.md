# DAG Evaluation Visualization Tool

Streamlit-based visualization tool for DAG evaluation results.

## Features

1. **Model Selection**: Select different models from the sidebar (reads from `DAG_eval/cleaned` directory)
   - File format: `match_results_{true/false}__{model_name}__v5_merged.jsonl`
   - `true/false` indicates whether it is multimodal evaluation

2. **RPF Distribution Plot (by Answer Correctness)**: 
   - Score is a binary value (0=incorrect, 1=correct)
   - Use box plot + scatter plot to show RPF distribution for correct/incorrect answers
   - Visualize the correlation between RPF and answer correctness
   - Hover to view UUID and detailed information
   - Statistics include: accuracy, average RPF, average RPF difference between correct/incorrect answers

3. **Question Navigation**:
   - **Click to jump**: Click points in the scatter plot to navigate directly
   - **UUID input box**: Copy UUID from the chart and paste to jump (supports partial matching)
   - **Previous/Next buttons**: Quick browsing
   - Display current progress (X / total)

4. **DAG Visualization** (wide display):
   - Ground Truth DAG and LLM DAG displayed in separate tabs
   - Node color coding:
     - 🟢 Light Green: Full match
     - 🟡 Yellow: Partial match (incomplete logic)
     - 🔴 Red: No match
     - ⚪ Gray: Non-critical nodes
     - 🩷 Pink: LLM hallucination nodes

## RPF Calculation Method

RPF (Reasoning Path Fidelity) calculation:

```
1. Iterate through each node in Ground Truth graph
2. For each node:
   - Unmatched → score = 0
   - Matched → calculate logic_ratio
     - Check if each parent node's corresponding LLM node can reach any LLM node of current node
     - logic_ratio = valid parent nodes / total parent nodes (1 if no parents)
   - node_score = points × logic_ratio
3. RPF = Σ(node_score) / Σ(points)
```

## Dependencies

```bash
pip install streamlit streamlit-agraph plotly pandas networkx
```

## Run

```bash
cd ./SUPERChem_eval/DAG_eval/view
streamlit run dag_viewer.py
```

Or specify port:

```bash
streamlit run dag_viewer.py --server.port 8501
```
