#!/bin/bash

#SBATCH --job-name=translation
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
#SBATCH --partition=rtx8000

DATA_DIR='../dataset/machine_translation'
DATA_PREFIX='iate.414'

MODEL_RECOVER_PATH='Helsinki-NLP/opus-mt-de-en'
OUTPUT_FILE='slurm/result2.out'


singularity exec --nv --bind /scratch/yx3038 --overlay /scratch/yx3038/overlay-25GB-500K.ext3:ro /scratch/yx3038/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate hug
cd /scratch/yx3038/Research/a_star_neurologic/translation
python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.de --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${DATA_PREFIX}.json \
  --batch_size 64 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
  --look_ahead_step 35  --alpha 0.05 --look_ahead_width 1"
