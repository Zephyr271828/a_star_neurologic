#!/bin/bash

#SBATCH --job-name=q_gen
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
#SBATCH --partition=a100_1,a100_2,h100_1

DATA_DIR='../dataset/question_generation'
# SPLIT='dev'
MODEL_RECOVER_PATH='gpt2-large'
OUTPUT_FILE='slurm/result4.out'


source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate /scratch/yx3038/penv
cd /scratch/yx3038/Research/a_star_neurologic/question_generation
python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1