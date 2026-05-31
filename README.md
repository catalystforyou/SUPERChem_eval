<div align="center">

# SUPERChem: A Multimodal Reasoning Benchmark in Chemistry

🌐 [Website](https://superchem.pku.edu.cn) | 📄 [Paper](https://arxiv.org/abs/2512.01274) | 🤗 [Dataset](https://huggingface.co/datasets/ZehuaZhao/SUPERChem)

</div>

This repository contains the official evaluation framework for **SUPERChem**, an expert-curated, reasoning-intensive multimodal benchmark for the rigorous evaluation of deep chemical reasoning in Large Language Models (LLMs) and Multimodal LLMs (MLLMs).

**License:** [MIT](LICENSE)

---

## Quick demo (recommended first step)

Verify your environment with bundled sample data (Gemini 2.5 Pro answers, no API key):

```bash
pip install -r requirements.txt
python demo/run_demo.py
```

See [demo/README.md](demo/README.md) for file descriptions and DAG_eval usage with the same sample.

---

## 1. System requirements

### Software dependencies

Install from the repository root:

```bash
pip install -r requirements.txt
```

| Component | Version (tested) |
|-----------|------------------|
| Python | 3.10, 3.11, 3.12 |
| pandas | 2.x |
| pyarrow | 14.x–21.x |
| openai | 1.x–2.x |
| PyYAML, loguru, tqdm | see `requirements.txt` |
| networkx, matplotlib | for `DAG_eval/` |
| plotly, scipy, seaborn, Pillow | for `analysis/` |
| streamlit | for `DAG_eval/view/` (optional) |

### Operating systems tested

- Ubuntu 22.04 / 24.04 LTS
- macOS 14+ (Apple Silicon and Intel)

### Hardware

- **Demo / accuracy scripts:** standard desktop or laptop (CPU only).
- **Full benchmark inference (`eval/`):** network access to your LLM API; no GPU required in this repo.
- **DAG / RPF pipeline (`DAG_eval/`):** API access to judge models; optional external [molecule comparison service](https://github.com/tom832/chemdraw-server) for structure matching (see `DAG_eval/README.md`).

---

## 2. Installation

```bash
git clone <repository-url>
cd SUPERChem_eval
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp eval/config.yaml.sample eval/config.yaml    # then add API keys for full eval
```

**Typical install time:** 2–5 minutes on a normal desktop (depends on network speed).

---

## 3. Demo

### Run the bundled demo

```bash
python demo/run_demo.py
```

| Item | Value |
|------|--------|
| Data | 10 questions + Gemini 2.5 Pro (text-only, high) answers in `demo/` |
| Expected output | Printed pass@1 accuracy (~50%, 5/10) and per-UUID scores |
| Expected runtime | &lt; 5 s (after `pip install`) |

### Demo contents

- `demo/questions_demo.parquet` — questions
- `demo/20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl` — model outputs
- `demo/ground_truth_graphs_detail.jsonl` — expert reasoning graphs for RPF

---

## 4. Instructions for use

### Full dataset

The complete benchmark (500 items) is on Hugging Face: [ZehuaZhao/SUPERChem](https://huggingface.co/datasets/ZehuaZhao/SUPERChem). Place downloaded files under `data/` following names in `eval/eval.sh` and `DAG_eval/README.md`.

### Generate model answers (`eval/`)

1. Copy `eval/config.yaml.sample` → `eval/config.yaml` and set API endpoints/keys.
2. Edit `eval/eval.sh` (model, `INPUT_FILE`, multimodal flag).
3. Run: `cd eval && bash eval.sh`  
   Outputs: `data/*.jsonl`.

Details: [eval/README.md](eval/README.md).

### Reasoning Path Fidelity / DAG evaluation (`DAG_eval/`)

1. Place questions parquet, model answers jsonl, and `ground_truth_graphs_detail.jsonl` under `DAG_eval/data/`.
2. Copy `DAG_eval/src/config.example.yaml` → `DAG_eval/src/config.yaml`.
3. Run `./run_full_pipeline.sh` or individual steps in `DAG_eval/src/`.

Details: [DAG_eval/README.md](DAG_eval/README.md).

### Analyze results (`analysis/`)

Process `data/*.jsonl` with scripts in `analysis/` (e.g. `calc_pass_withbaseline.py`, `draw_radar_plotly.py`). Figures go to `results/`.

Details: [analysis/README.md](analysis/README.md).

### (Optional) Reproducing paper figures

1. Obtain model answer files for the models reported in the paper (via `eval/` or released artifacts).
2. Run `analysis/calc_pass_withbaseline.py` for accuracy tables.
3. Run plotting scripts (`draw_radar_plotly.py`, `pass_k_curve.py`, etc.) with paths pointing to your `data/` files.

Exact figure-to-script mapping may vary by revision; use filenames in `results/` as a reference for expected outputs.

---

## Updates & News

* **[2026-03-16]** SUPERChem is adopted by [**MiroThinker-1.7**](https://arxiv.org/pdf/2603.15726).
* **[2026-02-14]** SUPERChem is adopted by [**ByteDance's Seed-2.0**](https://github.com/ByteDance-Seed/Seed2.0/blob/master/Seed2.0%20Model%20Card.pdf).
* **[2025-12-06]** **PDF Preview Released**: We have released the PDF version of SUPERChem in both English and Chinese to facilitate easier previewing and manual inspection, especially for non-technical users. You can download [SUPERChem-500.zip](https://huggingface.co/datasets/ZehuaZhao/SUPERChem/blob/main/SUPERChem-500.zip) to access the dataset in PDF format. The password to unzip the file is `SUPERChem2025`.

---

## Abstract

Current benchmarks for evaluating the chemical reasoning capabilities of Large Language Models (LLMs) are limited by oversimplified tasks, ceiling effects, lack of process-level evaluation, and misalignment with expert-level chemistry skills. To address these issues, we introduce **SUPERChem**, a benchmark of 500 expert-curated reasoning-intensive chemistry problems, covering diverse subfields and provided in both multimodal and text-only formats. Original content and an iterative curation pipeline eliminate flawed items and mitigate data contamination. Each problem is paired with an expert-authored solution path, enabling *Reasoning Path Fidelity* (RPF) scoring to evaluate reasoning quality beyond final-answer accuracy. Evaluations against a human baseline of 40.3% accuracy show that even the best-performing model, GPT-5 (High), reaches only 38.5%. By combining high difficulty, controlled multimodality, and process-level metrics, SUPERChem provides a rigorous platform for diagnosing and advancing AI chemical reasoning toward expert-level scientific inquiry.

---

## Key Features

- **Expert-Level Challenge**: 500 reasoning-intensive problems curated by domain experts.
- **Process-Level Evaluation**: **Reasoning Path Fidelity (RPF)** via expert solution DAGs.
- **Controlled Multimodality**: Text-only and multimodal variants per question.
- **Fine-Grained Ability Taxonomy**: Tags for knowledge and reasoning skills.
- **Contamination Resistant**: Expert-authored or non-public sources with human curation.

---

## Repository structure

```
.
├── demo/               # Small sample dataset + run_demo.py (start here)
├── eval/               # LLM answer generation
├── DAG_eval/           # DAG extraction, matching, RPF scoring
├── data/               # Full benchmark data and evaluation outputs
├── analysis/           # Metrics and plots
├── results/            # Generated figures
├── requirements.txt
└── LICENSE             # MIT
```

---

## Citation

If you use SUPERChem or this evaluation framework in your research, please cite our paper:

```bibtex
@misc{zhao2025superchemmultimodalreasoningbenchmark,
      title={SUPERChem: A Multimodal Reasoning Benchmark in Chemistry},
      author={Zehua Zhao and Zhixian Huang and Junren Li and Siyu Lin and Junting Zhou and Fengqi Cao and Kun Zhou and Rui Ge and Tingting Long and Yuexiang Zhu and Yan Liu and Jie Zheng and Junnian Wei and Rong Zhu and Peng Zou and Wenyu Li and Zekai Cheng and Tian Ding and Yaxuan Wang and Yizhao Yan and Tingru Wei and Haowei Ming and Weijie Mao and Chen Sun and Yiming Liu and Zichen Wang and Zuo Zhang and Tong Yang and Hao Ma and Zhen Gao and Jian Pei},
      year={2025},
      eprint={2512.01274},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.01274},
}
```
