#!/bin/bash

#SBATCH --job-name=cmg_s
#SBATCH --output=slurm/output.out
#SBATCH --error=slurm/error.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end           
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue

DATA_DIR='../dataset/commongen'
SPLIT='dev'
MODEL_RECOVER_PATH='fine-tuned_gpt2/checkpoint-14500'
OUTPUT_FILE='slurm/result2.out'


source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate scratch/yx3038/penv
# conda activate hug
cd /scratch/yx3038/Research/a_star_neurologic/commongen_supervised
python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 50 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.25 --look_ahead_width 1
