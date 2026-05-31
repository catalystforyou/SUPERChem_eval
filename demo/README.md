# SUPERChem demo data

This folder contains a **small, runnable sample** for reviewers and new users. It is derived from the public release split of SUPERChem and pre-computed answers from **Gemini 2.5 Pro** (text-only, `high` reasoning effort).

## Files

| File | Description |
|------|-------------|
| `questions_demo.parquet` | 10 chemistry questions (multimodal fields included) |
| `20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl` | Model answers and scores for those questions |
| `ground_truth_graphs_detail.jsonl` | Expert reasoning DAGs for RPF / `DAG_eval` |
| `dataset_split_map.json` | Split metadata for the 10 demo UUIDs |
| `20251015_baseline_demo.csv` | Human baseline rows for the same UUIDs |
| `run_demo.py` | No-API script that prints demo accuracy |

## Quick start (no API key)

From the repository root:

```bash
pip install -r requirements.txt
python demo/run_demo.py
```

**Expected output (approximate):** pass@1 accuracy about **50% (5/10)** on the bundled Gemini 2.5 Pro answers; human baseline printed for the same items. Runtime on a normal desktop: **under 5 seconds** after dependencies are installed.

## Using demo data with `DAG_eval` (API required)

Copy or symlink demo files into `DAG_eval/data/`, then configure `DAG_eval/src/config.yaml` and run matching on a subset:

```bash
mkdir -p DAG_eval/data
cp demo/questions_demo.parquet DAG_eval/data/20251014164938_questions.parquet
cp demo/ground_truth_graphs_detail.jsonl DAG_eval/data/
cp demo/20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl DAG_eval/data/

# Example: match DAG for 2 questions (requires API keys in config.yaml)
cd DAG_eval
python src/match_dag.py \
  --questions data/20251014164938_questions.parquet \
  --answers data/20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl \
  --ground-truth data/ground_truth_graphs_detail.jsonl \
  --output raw/demo_match_results.jsonl \
  --prompt prompts/match_prompt_v5.md \
  --model <your-judge-model> \
  --language en \
  --limit 2 \
  --workers 1
```

Full RPF pipeline: see `DAG_eval/README.md`.
