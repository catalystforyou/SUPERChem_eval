#!/bin/bash
set -euo pipefail

# Full pipeline:
# 1) match_dag.py
# 2) ability tagging with deepseek-chat
# 3) ability tag filter (keep uuid + parsed)
# 4) loop: validate -> rematch (poor+fair) until no "poor"

cd "$(dirname "$0")"

# ======== Config ========
VERSION="5"
##########################
#======key variables======
##########################
MODEL="" # LLM output model name
MULTIMODAL="false"

# Judge Models
MATCH_MODEL=""   # model for match_dag.py
AUX_MODEL=""                      # model for tagging/validation/rematch
RUN_TAG="${MODEL}__${MULTIMODAL}_${MATCH_MODEL}-v${VERSION}"

SRC_DIR="./src"
DATA_DIR="./data"
PROMPTS="./prompts"
RAW_OUTPUT_DIR="./raw"

# Create run directory
RUN_DIR="${RAW_OUTPUT_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"



# Data paths
QUESTIONS="${DATA_DIR}/20251014164938_questions.parquet"
ANSWERS="${DATA_DIR}/20251014164938_questions_release_en_${MULTIMODAL}__${MODEL}__1_0_1.jsonl"
GROUND_TRUTH="${DATA_DIR}/ground_truth_graphs_detail.jsonl"
ANSWER_FIELD="llm_output"
LANGUAGE="en"

# Prompts
MATCH_PROMPT="${PROMPTS}/match_prompt_v${VERSION}.md"
TAG_PROMPT="${PROMPTS}/ability_tagging_prompt.md"
REMATCH_PROMPT="${PROMPTS}/rematch_prompt.md"
VALIDATION_PROMPT="${PROMPTS}/validation_prompt.md"

# Match output (all in RUN_DIR)
MATCH_OUTPUT="${RUN_DIR}/match_results.jsonl"
TAGGED_OUTPUT="${RUN_DIR}/match_results_tagged.jsonl"
TAGGED_FILTERED_OUTPUT="${RUN_DIR}/match_results_tagged_output.jsonl"

# Match params
MATCH_WORKERS=50
MATCH_STAGGER_DELAY=1
MATCH_MAX_RETRIES=5
MATCH_TIMEOUT=300
MATCH_TEMPERATURE=0.8

# Tagging params
TAG_WORKERS=64
TAG_MAX_RETRIES=3
TAG_TIMEOUT=300
TAG_TEMPERATURE=0.8

# Validation params
VALID_SAMPLE_SIZE=500
VALID_WORKERS=64
VALID_STAGGER_DELAY=1
VALID_TIMEOUT=300
VALID_TEMPERATURE=0.8

# Rematch params
REMATCH_WORKERS=64
REMATCH_STAGGER_DELAY=1
REMATCH_TIMEOUT=300
REMATCH_TEMPERATURE=0.8
REMATCH_MAX_RETRIES=5
REMATCH_ONLY_QUALITY="fair,poor"

# Loop guard
MAX_ROUNDS=10

# ======== Step 1: match_dag.py ========
echo "==> Step 1/6: match_dag.py with ${MATCH_MODEL}"
python "$SRC_DIR/match_dag.py" \
  --questions "$QUESTIONS" \
  --answers "$ANSWERS" \
  --ground-truth "$GROUND_TRUTH" \
  --output "$MATCH_OUTPUT" \
  --model "$MATCH_MODEL" \
  --language "$LANGUAGE" \
  --answer-field "$ANSWER_FIELD" \
  --prompt "$MATCH_PROMPT" \
  --max-retries "$MATCH_MAX_RETRIES" \
  --timeout "$MATCH_TIMEOUT" \
  --temperature "$MATCH_TEMPERATURE" \
  --workers "$MATCH_WORKERS" \
  --stagger-delay "$MATCH_STAGGER_DELAY" \
  --resume

# ======== Step 1.5: Clean null parsed records and retry ========
echo "==> Step 1.5/6: Checking for null parsed records"
python "$SRC_DIR/clean_null_parsed.py" --input "$MATCH_OUTPUT"

# Check if there are still records to process (after cleaning)
total_records=$(wc -l < "$MATCH_OUTPUT" 2>/dev/null || echo "0")
if [[ "$total_records" -gt 0 ]]; then
  echo "Retrying match_dag.py to process cleaned records (resume mode)"
  python "$SRC_DIR/match_dag.py" \
    --questions "$QUESTIONS" \
    --answers "$ANSWERS" \
    --ground-truth "$GROUND_TRUTH" \
    --output "$MATCH_OUTPUT" \
    --model "$MATCH_MODEL" \
    --language "$LANGUAGE" \
    --answer-field "$ANSWER_FIELD" \
    --prompt "$MATCH_PROMPT" \
    --max-retries "$MATCH_MAX_RETRIES" \
    --timeout "$MATCH_TIMEOUT" \
    --temperature "$MATCH_TEMPERATURE" \
    --workers "$MATCH_WORKERS" \
    --stagger-delay "$MATCH_STAGGER_DELAY" \
    --resume
  
  # Check again after retry
  echo "Final check for null parsed records"
  python "$SRC_DIR/clean_null_parsed.py" --input "$MATCH_OUTPUT"
fi

# ======== Step 2: ability tagging (deepseek-chat) ========
echo "==> Step 2/6: ability tagging with ${AUX_MODEL}"
python "$SRC_DIR/ability_tag_model_answer_nodes.py" \
  --match-results "$MATCH_OUTPUT" \
  --output "$TAGGED_OUTPUT" \
  --model "$AUX_MODEL" \
  --prompt-template "$TAG_PROMPT" \
  --max-retries "$TAG_MAX_RETRIES" \
  --timeout "$TAG_TIMEOUT" \
  --temperature "$TAG_TEMPERATURE" \
  --workers "$TAG_WORKERS" \
  --resume

# ======== Step 3: ability tag filter ========
echo "==> Step 3/6: ability tag filter"
python "$SRC_DIR/ability_tag_filter_fields.py" \
  --input "$TAGGED_OUTPUT" \
  --output "$TAGGED_FILTERED_OUTPUT"

# ======== Step 4: validate + rematch loop ========
echo "==> Step 4/6: validate + rematch loop"

# Always start from round 1; resume handled by Python scripts
current_match_file="$TAGGED_FILTERED_OUTPUT"
round=1

echo "Starting from round ${round} (resume handled by Python scripts)"

while true; do
  if [[ "$round" -gt "$MAX_ROUNDS" ]]; then
    echo "Reached MAX_ROUNDS=${MAX_ROUNDS}. Stopping."
    break
  fi

  validation_dir="${RUN_DIR}/validation_reports_round${round}"
  validation_details="${validation_dir}/validation_details.jsonl"
  
  mkdir -p "$validation_dir"
  echo "---- Round ${round}: validate (resume) ----"
  python "$SRC_DIR/validate_match_quality.py" \
    --match-results "$current_match_file" \
    --questions "$QUESTIONS" \
    --answers "$ANSWERS" \
    --ground-truth "$GROUND_TRUTH" \
    --output-dir "$validation_dir" \
    --model "$AUX_MODEL" \
    --language "$LANGUAGE" \
    --prompt-template "$VALIDATION_PROMPT" \
    --sample-size "$VALID_SAMPLE_SIZE" \
    --max-retries "$REMATCH_MAX_RETRIES" \
    --timeout "$VALID_TIMEOUT" \
    --temperature "$VALID_TEMPERATURE" \
    --workers "$VALID_WORKERS" \
    --stagger-delay "$VALID_STAGGER_DELAY" \
    --resume

  validation_details="${validation_dir}/validation_details.jsonl"
  if [[ ! -f "$validation_details" ]]; then
    echo "ERROR: validation_details.jsonl not found: $validation_details"
    break
  fi

  poor_count="$(python3 - <<'PY' "$validation_details"
import json
import sys
path = sys.argv[1]
poor = 0
total = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("validation_status") != "completed":
            continue
        total += 1
        if rec.get("validation_result", {}).get("overall_quality") == "poor":
            poor += 1
print(poor)
PY
)"

  echo "Round ${round}: poor_count=${poor_count}"
  if [[ "$poor_count" -eq 0 ]]; then
    echo "No poor results found. Loop finished."
    break
  fi

  next_match_file="${RUN_DIR}/match_results_rematch_round${round}.jsonl"
  
  echo "---- Round ${round}: rematch (poor+fair, resume) ----"
  python "$SRC_DIR/rematch_dag.py" \
    --tagged-output "$current_match_file" \
    --validation-details "$validation_details" \
    --questions "$QUESTIONS" \
    --answers "$ANSWERS" \
    --ground-truth "$GROUND_TRUTH" \
    --output "$next_match_file" \
    --model "$AUX_MODEL" \
    --language "$LANGUAGE" \
    --answer-field "$ANSWER_FIELD" \
    --prompt-template "$REMATCH_PROMPT" \
    --max-retries "$REMATCH_MAX_RETRIES" \
    --timeout "$REMATCH_TIMEOUT" \
    --temperature "$REMATCH_TEMPERATURE" \
    --workers "$REMATCH_WORKERS" \
    --stagger-delay "$REMATCH_STAGGER_DELAY" \
    --only-quality-threshold "$REMATCH_ONLY_QUALITY" \
    --resume

  current_match_file="$next_match_file"
  round=$((round + 1))
done

# ======== Step 5: Merge all rematch results ========
echo "==> Step 5/6: Merge rematch results"
MERGED_OUTPUT="${RUN_DIR}/match_results_merged.jsonl"

python "$SRC_DIR/merge_rematch_results.py" \
  --input-dir "$RUN_DIR" \
  --output "$MERGED_OUTPUT" \
  --log-level INFO

# ======== Step 6: Copy -> clean -> add score (overwrite) ========
echo "==> Step 6/6: Copy to raw, clean, add score"
RAW_DIR="./raw"
CLEANED_DIR="./cleaned"
RAW_OUTPUT="${RAW_DIR}/match_results_${MULTIMODAL}__${MODEL}__v${VERSION}_merged.jsonl"
CLEANED_OUTPUT="${CLEANED_DIR}/match_results_${MULTIMODAL}__${MODEL}__v${VERSION}_merged.jsonl"

mkdir -p "$RAW_DIR" "$CLEANED_DIR"
cp "$MERGED_OUTPUT" "$RAW_OUTPUT"

python - <<'PY' "$RAW_OUTPUT" "$CLEANED_OUTPUT"
import json
import sys

raw_path = sys.argv[1]
cleaned_path = sys.argv[2]

with open(raw_path, "r", encoding="utf-8") as infile, open(
    cleaned_path, "w", encoding="utf-8"
) as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        parsed = data.get("parsed") or {}
        cleaned_data = {
            "uuid": data.get("uuid"),
            **parsed,
        }
        nodes = cleaned_data.get("nodes", [])
        for node in nodes:
            node_id = node.get("id", "")
            if not node_id.startswith("H") or len(node_id) <= 1:
                print("Warning: Node ID does not start with 'H':", node_id)
        outfile.write(json.dumps(cleaned_data, ensure_ascii=False) + "\n")

print(f"Data cleaning completed, saved to {cleaned_path}")
PY

CLEANED_TMP="${CLEANED_OUTPUT}.tmp"
python "$SRC_DIR/add_score.py" \
  --answer "$ANSWERS" \
  --judge "$CLEANED_OUTPUT" \
  --output "$CLEANED_TMP"
mv "$CLEANED_TMP" "$CLEANED_OUTPUT"
rm "$RAW_OUTPUT"

echo ""
echo "✅ Pipeline completed!"
echo "📂 All results saved in: ${RUN_DIR}/"
echo "📄 Merged results: ${MERGED_OUTPUT}"
