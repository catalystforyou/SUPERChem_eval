#!/bin/bash

# Configuration
INPUT_FILE="../data/20251014164938_questions.parquet"
OUTPUT_PREFIX="${INPUT_FILE%.parquet}"
LANGUAGE="en"
MULTIMODAL=true  # Set to true or false
N_PROCS=4
N_THREADS=10

# Model configurations: "model_name|temperature|passk|reasoning_effort"
MODELS=(
    # "o3-2025-04-16|1.0|8|high"
    # "o4-mini|1.0|8|high"
    # "gpt-4o-2024-11-20|1.0|8|"
    "gemini-2.5-pro|1.0|1|high"
    # "deepseek-v3.1-128k|1.0|8|"
    # "deepseek-v3.1-thinking-128k|1.0|8|"
    # "qwen3-235b-a22b-thinking-2507|1.0|8|"
    # "qwen3-235b-a22b-instruct-2507|1.0|8|"
    # "gpt-5|1.0|8|high"
    # "gpt-5|1.0|8|medium"
    # "gpt-5|1.0|8|low"
    # "gpt-5|1|8|mini"
)

# Function to validate and convert boolean values
validate_bool() {
    local input="$1"
    case "${input,,}" in  # ${input,,} Transforms input to lowercase
        true|1)
            echo "True"  # Python recognizes this as True
            ;;
        false|0)
            echo ""      # An empty string is recognized as False in Python
            ;;
        *)
            echo "Error: Invalid boolean value '$input'. Expected: true, True, 1, false, False, or 0" >&2
            exit 1
            ;;
    esac
}
 
# Validate and convert MULTIMODAL parameter
MULTIMODAL_ARG=$(validate_bool "$MULTIMODAL")

# Process each model configuration
for config in "${MODELS[@]}"; do
    IFS='|' read -r model temp pass_k reasoning_effort <<< "$config"

    # Sanitize model name for filename
    safe_model=$(echo "$model" | sed 's/\//-/g; s/\./_/g')
    safe_temp=$(echo "$temp" | sed 's/\./_/g')
    
    # Generate output filename
    output_file="data/ckpt_${OUTPUT_PREFIX}_${LANGUAGE}_${MULTIMODAL}_${safe_model}_${reasoning_effort}_${safe_temp}_${pass_k}.jsonl"

    echo "Processing $model with temperature=$temp, pass_k=$pass_k, reasoning_effort=$reasoning_effort"
    echo "Output: $output_file"
    
    # Build command
    cmd="python eval_ckpt.py \
        --input \"$INPUT_FILE\" \
        --output \"$output_file\" \
        --model \"$model\" \
        --temperature \"$temp\" \
        --pass-k \"$pass_k\" \
        --language \"$LANGUAGE\" \
        --multimodal \"$MULTIMODAL_ARG\" \
        --n-procs \"$N_PROCS\" \
        --n-threads \"$N_THREADS\""

    # Add reasoning-effort parameter only if it's not empty
    if [[ -n "$reasoning_effort" ]]; then
        cmd="$cmd --reasoning-effort \"$reasoning_effort\""
    fi
    
    # Execute the command
    eval $cmd
    
    echo "Completed $model"
    echo "---"
done

echo "All models completed!"