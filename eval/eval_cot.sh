#!/bin/bash

# --- Configuration ---

# 1. Path to the file with original questions and ground truth analysis.
#    This file is the same for all evaluation runs.
QUESTIONS_FILE="../data/20251014164938_questions.parquet"

# 2. The single, powerful model that will act as the evaluator/judge.
EVALUATOR_MODEL="gemini-2.5-pro" 

# 3. The prefix used for all model answer files. The script will find all
#    files starting with this prefix. This should match the prefix from your
#    original answer generation script.
ANSWERS_PREFIX="../data/${QUESTIONS_FILE%.parquet}"

# 4. General settings for the python evaluation script.
LANGUAGE="en"
N_PROCS=4
N_THREADS=10
EVALUATOR_TEMPERATURE=1.0 # Temperature for the evaluator model

# --- Script Logic ---

echo "Starting automated CoT evaluation..."
echo "Using single evaluator model: $EVALUATOR_MODEL"
echo "Searching for answer files with prefix: ${ANSWERS_PREFIX}_*.jsonl"
echo "================================================="

# Find all .jsonl files that match the answer prefix.
# The glob will expand to a list of all matching filenames.
for answer_file_to_evaluate in ${ANSWERS_PREFIX}_*.jsonl; do

    # if "EVAL" in filename, then skip
    if [[ "$answer_file_to_evaluate" == *"_EVAL_BY_"* ]]; then
        echo "Skipping already evaluated file: $answer_file_to_evaluate"
        echo "---"
        continue # continue命令会跳过本次循环，处理下一个文件
    fi
    
    # --- Safety Checks ---
    # a) Check if the glob found any files at all.
    if [ ! -f "$answer_file_to_evaluate" ]; then
        echo "Error: No answer files found matching the pattern '${ANSWERS_PREFIX}_*.jsonl'."
        echo "Please ensure the answer files are in this directory and the prefix is correct."
        exit 1
    fi

    if [[ "$answer_file_to_evaluate" == *"_EVAL_BY_"* ]]; then
        echo "Skipping already evaluated file: $answer_file_to_evaluate"
        echo "---"
        continue # continue命令会跳过本次循环，处理下一个文件
    fi
    
    echo "Found answer file: $answer_file_to_evaluate"

    # --- Generate a descriptive output filename for this evaluation run ---
    # This takes the original answer file name and inserts the evaluation info.
    # Example: input.jsonl -> input_EVAL_BY_gpt-5.jsonl
    safe_evaluator_model=$(echo "$EVALUATOR_MODEL" | sed 's/\//-/g; s/\./_/g')
    evaluation_output_file=$(echo "$answer_file_to_evaluate" | sed "s/\.jsonl/_EVAL_BY_${safe_evaluator_model}.jsonl/")
    
    echo "Evaluating..."
    echo "  -> Evaluator: $EVALUATOR_MODEL"
    echo "  -> Output will be saved to: data/$evaluation_output_file"
    
    # --- Build and execute the command ---
    # The python script is assumed to be named 'eval_cot.py'
    cmd="python eval_cot.py \
        --input \"$QUESTIONS_FILE\" \
        --answers-input \"$answer_file_to_evaluate\" \
        --output \"data/$evaluation_output_file\" \
        --model \"$EVALUATOR_MODEL\" \
        --temperature \"$EVALUATOR_TEMPERATURE\" \
        --language \"$LANGUAGE\" \
        --pass-k 1 \
        --n-procs \"$N_PROCS\" \
        --n-threads \"$N_THREADS\""
    
    # Execute the command
    eval $cmd
    
    echo "Completed evaluation for $answer_file_to_evaluate"
    echo "---"
done

echo "All found answer files have been evaluated!"