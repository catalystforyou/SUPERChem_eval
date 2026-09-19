# Release manifests

`release_inputs.candidate.json` resolves the locally prepared 25 configurations ×
3 independent answer runs. It applies the 12 GPT-5 Low/Medium replacements and the
two genuinely generated o3 rep03 batches, without retaining synthetic repeats.

This file is **candidate input provenance only**, not the manuscript final release:
all `matches` entries are null until returned final DS-GT outputs are reconciled.
Paths are relative to this manifest; referenced full files are local artifacts,
not bundled with a normal clone. For a self-contained example use `demo/manifest.json`.

```bash
python -m superchem validate --manifest configs/release_inputs.candidate.json
# Must fail until all final match files are supplied:
python -m superchem validate --manifest configs/release_inputs.candidate.json --require-matches
```

Maintainer-only regeneration (requires local historical/handoff artifacts):

```bash
python script/build_release_manifests.py --candidate
```

Manifest fields: schema_version, formula_version, state, ordered question_ids,
ground_truth `{path, sha256}`, and runs with config_id, replicate,
independent_generation, synthetic_resample, answers and optional matches.
One file per run must contain every question exactly once. Explicit failures are
rows with status=false; missing rows, duplicate UUIDs and mismatched hashes fail.
