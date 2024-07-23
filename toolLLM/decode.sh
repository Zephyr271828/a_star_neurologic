#!/usr/bin/env bash


DATA_DIR='../dataset/commongen'
SPLIT='dev'
# MODEL_RECOVER_PATH='/scratch/yx3038/tmp/llama3'
MODEL_RECOVER_PATH='gpt2'

INPUT_PATH='/scratch/yx3038/Research/StableToolBench/inference_results/constraints/queries.txt'
CONSTRAINT_FILE='/scratch/yx3038/Research/StableToolBench/inference_results/constraints/constraints.json'

DEVICES=0
OUTPUT_FILE=output_decode/result.txt



# --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
# --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
# --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --input_path ${INPUT_PATH} --output_file ${OUTPUT_FILE} \
  --constraint_file ${CONSTRAINT_FILE} \
  --key_constraint_file ${CONSTRAINT_FILE} \
  --batch_size 1 \
  --ngram_size 3 --length_penalty 0.2 --max_tgt_length 150 \
  --prune_factor 50 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.25 --look_ahead_width 1 #--fusion_t 1.0

# neurologic with sampling look-ahead
# CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
#   --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
#   --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
#   --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
#   --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
#   --ngram_size 3 --length_penalty 0.2 \
#   --prune_factor 50 --sat_tolerance 2 \
#   --look_ahead_step 5  --alpha 0.3  --look_ahead_sample --look_ahead_width 5

# neurologic with beam look-ahead
#CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
  # --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  # --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  # --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  # --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  # --ngram_size 3 --length_penalty 0.2 \
  # --prune_factor 50 --sat_tolerance 2 \
  # --look_ahead_step 5  --alpha 0.45 --look_ahead_width 4