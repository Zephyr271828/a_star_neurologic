#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/a_star_neurologic
# MODEL_NAME='/scratch/yx3038/tmp/llama3'
MODEL_NAME='gpt2'

INPUT_PATH='/scratch/yx3038/Research/StableToolBench/inference_results/constraints/queries.txt'
CONSTRAINT_FILE='/scratch/yx3038/Research/StableToolBench/inference_results/constraints/constraints.json'

DEVICES=0
OUTPUT_FILE=output_decode/result.txt

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode_gpt2.py  \
  --model_name ${MODEL_NAME} \
  --input_path ${INPUT_PATH} \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${CONSTRAINT_FILE} \
  --key_constraint_file ${CONSTRAINT_FILE} \
  --batch_size 16 --beam_size 20 --max_tgt_length 150 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 #--fusion_t 1.0
