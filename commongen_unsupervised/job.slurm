#!/bin/bash

#SBATCH --job-name=cmg_u
#SBATCH --output=slurm/output.out
#SBATCH --error=slurm/error.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end           
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --requeue
#SBATCH --partition=rtx8000

DATA_DIR='../dataset/commongen'
SPLIT='test'
MODEL_RECOVER_PATH='gpt2-large'
OUTPUT_FILE='slurm/test_result.out'


singularity exec --nv --bind /scratch/yx3038 --overlay /scratch/yx3038/overlay-25GB-500K.ext3:ro /scratch/yx3038/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate hug
cd /scratch/yx3038/Research/a_star_neurologic/commongen_unsupervised
python decode_gpt2.py --model_name ${MODEL_RECOVER_PATH} \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 16 --beam_size 12 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1"
