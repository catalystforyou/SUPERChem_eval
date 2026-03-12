# DAG Evaluation

A framework for evaluating Large Language Model (LLM) reasoning capabilities in chemistry using Directed Acyclic Graphs (DAGs).

## Overview

This project provides tools to:
1. Extract reasoning chains from LLM answers as DAGs
2. Match LLM reasoning graphs against ground truth graphs
3. Evaluate graph similarity using metrics like RPF (Reasoning Path Fidelity)
4. Validate and improve match quality through iterative refinement
5. Visualize evaluation results

## Directory Structure

```
DAG_eval/
├── dag_evaluation.ipynb    # Jupyter notebook for analysis and visualization
├── run_full_pipeline.sh  # Main pipeline script
├── src/                   # Python source code
│   ├── match_dag.py              # Extract and match DAG from LLM answers
│   ├── validate_match_quality.py # Validate match quality using LLM
│   ├── rematch_dag.py            # Re-match based on validation feedback
│   ├── merge_rematch_results.py  # Merge rematch results
│   ├── validate_dag.py           # DAG structure validation
│   ├── ability_tag_model_answer_nodes.py  # Tag reasoning abilities
│   ├── ability_tag_filter_fields.py       # Filter by ability tags
│   ├── add_score.py             # Add accuracy scores
│   ├── clean_null_parsed.py      # Clean null parsed records
│   └── config.example.yaml      # Configuration template
├── prompts/                # LLM prompt templates
│   ├── match_prompt_v5.md      # DAG matching prompt
│   ├── validation_prompt.md     # Validation prompt
│   ├── rematch_prompt.md        # Re-matching prompt
│   └── ability_tagging_prompt.md # Ability tagging prompt
├── view/                   # Visualization tools
│   ├── dag_viewer.py            # Streamlit-based viewer
│   └── README.md                # Viewer documentation
└── README.md              # This file
```

## Data Requirements

**Important**: Place the following data files in the `./data` directory before running the pipeline:

1. **Questions Dataset**:
   ```
   ./data/20251014164938_questions.parquet
   ```
   This file contains the chemistry questions with ground truth DAGs.

2. **LLM Answers**:
   ```
   ./data/20251014164938_questions_release_en_{multimodal}__{model}__1_0_1.jsonl
   ```
   - `{multimodal}`: `true` or `false`
   - `{model}`: LLM model name (e.g., `gpt-5_high`, `gemini-2_5-pro_high`)

   Example filename:
   - `20251014164938_questions_release_en_false__gpt-5_high__1_0_1.jsonl` (text-only)
   - `20251014164938_questions_release_en_true__gemini-2_5-pro_high__1_0_1.jsonl` (multimodal)

The pipeline expects these specific file paths as defined in `run_full_pipeline.sh`:
- `QUESTIONS="${DATA_DIR}/20251014164938_questions.parquet"`
- `ANSWERS="${DATA_DIR}/20251014164938_questions_release_en_${MULTIMODAL}__${MODEL}__1_0_1.jsonl"`

## Installation

```bash
pip install pandas pyyaml loguru openai tqdm requests networkx matplotlib
pip install streamlit streamlit-agraph plotly
```

## Configuration

Copy the example config and fill in your API keys:

```bash
cp src/config.example.yaml src/config.yaml
```

Edit `src/config.yaml` with your model configurations:
- `model_list`: List of LLM models with base_url and api_key
- `mol_compare`: Configuration for molecule comparison API

### Molecule Comparison Tool

The `batch_mol_compare` tool is used to verify whether chemical structures in the model answer match those in the ground truth. For setting up your own molecule comparison service, please refer to: https://github.com/tom832/chemdraw-server

## Usage

### Full Pipeline

Run the complete evaluation pipeline:

```bash
./run_full_pipeline.sh
```

Before running, configure the following variables in the script:
- `MODEL`: LLM model name to evaluate
- `MATCH_MODEL`: Model for DAG matching
- `AUX_MODEL`: Model for tagging/validation

### Individual Steps

1. **Extract and Match DAG**:
```bash
python src/match_dag.py --questions <questions> --answers <answers> \
  --ground-truth <ground_truth> --output <output> --model <model>
```

2. **Ability Tagging**:
```bash
python src/ability_tag_model_answer_nodes.py --match-results <input> \
  --output <output> --model <model>
```

3. **Validate Match Quality**:
```bash
python src/validate_match_quality.py --match-results <input> \
  --questions <questions> --answers <answers> --ground-truth <ground_truth> \
  --output-dir <output_dir> --model <model>
```

4. **Re-match Based on Validation**:
```bash
python src/rematch_dag.py --tagged-output <input> \
  --validation-details <validation_details> --questions <questions> \
  --answers <answers> --ground-truth <ground_truth> --output <output> \
  --model <model>
```

5. **Merge Results**:
```bash
python src/merge_rematch_results.py --input-dir <input_dir> --output <output>
```

### Visualization

Launch the Streamlit viewer:

```bash
cd view
streamlit run dag_viewer.py
```

## Evaluation Metrics

### Single Graph Metrics
- **Exploration Density**: Ratio of actual edges to maximum possible edges
- **Branching Factor**: Average excess branching of node out-degrees
- **Convergence Factor**: Average excess convergence of node in-degrees
- **Linearity**: Proportion of nodes with degree ≤ 2
- **Dangling Count**: Number of leaf nodes beyond the expected terminal node

### Match Quality Metrics
- **RPF (Reasoning Path Fidelity)**: Fidelity considering node coverage and path structure
- **Recall**: Degree to which ground truth nodes are correctly covered
- **Precision**: How many LLM nodes are valid
- **F1**: Harmonic mean of recall and precision

## Data Format

### Input Questions (Parquet/JSONL)
- `uuid`: Unique question identifier
- `question_en`: Question text (English)
- `options_en`: Multiple choice options
- `explanation_en`: Ground truth explanation
- `ground_truth_graph`: Ground truth DAG structure

### Input Answers (JSONL)
- `uuid`: Question identifier
- `llm_output`: LLM's answer text

### Output Match Results (JSONL)
- `uuid`: Question identifier
- `nodes`: Extracted reasoning nodes
- `edges`: Reasoning dependencies
- `matches`: Node-to-node mappings between LLM and ground truth

