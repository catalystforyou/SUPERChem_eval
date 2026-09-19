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
pip install -r requirements-offline.txt
python demo/run_demo.py
```

See [demo/README.md](demo/README.md) for file descriptions and DAG_eval usage with the same sample.

The demo now verifies frozen file hashes and reproduces **ACC, RPF, node-only,
branching factor and dangling count** from historical precomputed matches, without
API keys. It is a reproducibility example, not the final revised-paper leaderboard.

### Offline evaluation entry point

Run from the repository root:

```bash
python -m superchem validate --manifest demo/manifest.json --require-matches
python -m superchem metrics --manifest demo/manifest.json --output outputs/demo_metrics
python -m unittest discover -s tests -v
```

The metrics command refuses to overwrite an existing output directory. Outputs
include per-question metrics and per-run means/sample variances. See
[the release protocol](docs/release_protocol.md) for formulas, data versions and
failure handling. Legacy API generation scripts remain under `eval/`; archived
handoff packages are not modified by the offline entry point.

---

## 1. System requirements

### Software dependencies

For the broader legacy API/analysis tools, install from the repository root
(the offline demo only needs `requirements-offline.txt`):

```bash
pip install -r requirements.txt
```

| Component | Dependency scope |
|-----------|------------------|
| Python | Offline code supports 3.10+; local verification used 3.13.12 |
| pandas | Offline: 2.x–3.x; legacy full environment: 2.x |
| pyarrow | Offline: 14.x–24.x; legacy full environment: 14.x–21.x |
| openai | 1.x–2.x |
| PyYAML, loguru, tqdm | see `requirements.txt` |
| networkx, matplotlib | for `DAG_eval/` |
| plotly, scipy, seaborn, Pillow | for `analysis/` |
| streamlit | for `DAG_eval/view/` (optional) |

### Verified environment

The offline demo and tests were run on Linux x86_64, Python 3.13.12,
pandas 3.0.3/3.0.5, pyarrow 24.0.0 and networkx 3.6.1 (including a fresh,
isolated install from the local package cache). Python 3.10/3.12 Linux
jobs are configured in CI; their configuration is not proof of a completed run.
macOS/Windows and the complete API/visualization environment were not retested
in this release review. `requirements-offline.txt` is sufficient for the demo;
`requirements.txt` retains the broader legacy API/analysis/viewer dependencies.

### Hardware

- **Demo / accuracy scripts:** standard desktop or laptop (CPU only).
- **Full benchmark inference (`eval/`):** network access to your LLM API; no GPU required in this repo.
- **Offline DAG scoring:** CPU only, no network/API required.
- **API DAG matching (`DAG_eval/`):** judge access and either the external [ChemDraw service](https://github.com/tom832/chemdraw-server) or local **OPSIN + RDKit**. The open-source backend needs a Java JDK for setup, but no molecular-service key. See [backend setup and switching](docs/mol_compare.md).

---

## 2. Installation

```bash
git clone https://github.com/tom832/SUPERChem_eval.git
cd SUPERChem_eval
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-offline.txt
python demo/run_demo.py
# Optional: install the broader legacy API/analysis environment separately.
# pip install -r requirements.txt
# cp eval/config.yaml.sample eval/config.yaml
```

Installation time depends on network access and wheel availability; a fresh
network installation was not timed in this review.

---

## 3. Demo

### Run the bundled demo

```bash
python demo/run_demo.py
```

| Item | Value |
|------|--------|
| Data | 10 questions + Gemini 2.5 Pro (text-only, high) answers in `demo/` |
| Expected output | ACC 50% (5/10), RPF 0.516569, node-only 0.599534, BF 0.220613, DC 0 |
| Expected runtime | Seconds for 10 questions after installation; hardware-dependent |

### Demo contents

- `demo/questions_demo.parquet` — questions
- `demo/20251014164938_questions_release_en_false__gemini-2_5-pro_high__1_0_1.jsonl` — model outputs
- `demo/ground_truth_graphs_detail.jsonl` — expert reasoning graphs for RPF
- `demo/precomputed_matches.jsonl` — historical matches for exactly these answers/GT
- `demo/manifest.json`, `demo/expected_metrics.json` — input hashes and regression values

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

For local open-source chemical name/structure comparison:

```bash
pip install -r requirements-mol.txt
python script/install_opsin.py
python -m superchem mol-compare --backend opsin --mol1 ethanol --mol2 CCO
```

Set `mol_compare.backend: opsin` in the YAML or pass
`--mol-compare-backend opsin` to `DAG_eval/src/match_dag.py`.
`chemdraw` remains available and is the backward-compatible default.
OPSIN failures/ambiguity are reported as unknown, not unequal. See
[comparison policy, limits and tests](docs/mol_compare.md).

**Protocol note:** the shell pipeline below is the historical semantic
validation/rematch workflow. The revision handoff workflow combines extraction
and matching in one call, followed by up to three recovery rounds for failed or
structurally invalid outputs. These are distinct protocols. For offline scoring,
use `python -m superchem metrics`; for final manuscript configuration status,
see [docs/release_protocol.md](docs/release_protocol.md).

1. Place questions parquet, model answers jsonl, and `ground_truth_graphs_detail.jsonl` under `DAG_eval/data/`.
2. Copy `DAG_eval/src/config.example.yaml` → `DAG_eval/src/config.yaml`.
3. Run `cd DAG_eval && bash run_full_pipeline.sh` or individual steps in `DAG_eval/src/`.

Details: [DAG_eval/README.md](DAG_eval/README.md).

### Analyze results (`analysis/`)

Process `data/*.jsonl` with scripts in `analysis/` (e.g. `calc_pass_withbaseline.py`, `draw_radar_plotly.py`). Figures go to `results/`.

Details: [analysis/README.md](analysis/README.md).

### (Optional) Reproducing paper figures

1. Obtain model answer files for the models reported in the paper (via `eval/` or released artifacts).
2. Run `analysis/calc_pass_withbaseline.py` for accuracy tables.
3. Run plotting scripts (`draw_radar_plotly.py`, `pass_k_curve.py`, etc.) with paths pointing to your `data/` files.

Final revised figure-to-input mappings are pending reconciliation with the final
returned matches; historical files in `results/` are not a final release manifest.

### Release hygiene

Current human baseline files use consistent `participant_###` pseudonyms. Old
Git objects and archived deliveries still retain the original identifiers;
do not distribute `.git/`, local API configs, or the complete research workspace
as a sanitized release. A read-only current-file check is available:

```bash
python script/check_release_safety.py --report outputs/release_review/safety_scan.json
```

It reports locations/counts, never matched secret values. The scan is heuristic
and does not replace review of image/PDF contents or historical data.
See [the review record](docs/release_security_review.md) for scope and test results.

---

## Updates & News

* **[2026-03-16]** SUPERChem is adopted by [**MiroThinker-1.7**](https://arxiv.org/pdf/2603.15726).
* **[2026-02-14]** SUPERChem is adopted by [**ByteDance's Seed-2.0**](https://github.com/ByteDance-Seed/Seed2.0/blob/master/Seed2.0%20Model%20Card.pdf).
* **[2025-12-06]** **PDF Preview Released**: We have released the PDF version of SUPERChem in both English and Chinese to facilitate easier previewing and manual inspection, especially for non-technical users. You can download [SUPERChem-500.zip](https://huggingface.co/datasets/ZehuaZhao/SUPERChem/blob/main/SUPERChem-500.zip) to access the dataset in PDF format. The password to unzip the file is `SUPERChem2025`.

---

## Abstract

**SUPERChem** contains 500 expert-curated chemistry reasoning problems in text-only
and multimodal formats. Expert-authored checkpoints support process-level
evaluation with *Reasoning Path Fidelity* (RPF), alongside final-answer accuracy.
The benchmark evaluates long-chain chemistry problem solving; it does not directly
measure autonomous scientific discovery. Numerical rankings for the revised paper
will be linked to a final, versioned input-and-result manifest after reconciliation.

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
