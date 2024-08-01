#!/bin/bash

USER=yx3038
CONDA_ENV=/vast/yx3038/penv/
TASK_NAME=decode
FILE_NAME=decode
TMP_SCRIPT=$(mktemp $(pwd)/slurm_job_XXXXXX.slurm)


cat <<EOL > $TMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=${TASK_NAME}
#SBATCH --output=$(pwd)/output_slurm/${TASK_NAME}.out
#SBATCH --error=$(pwd)/output_slurm/${TASK_NAME}.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=${USER}@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd $(pwd)

EOL

grep -v "^#" ./scripts/${FILE_NAME}.sh >> $TMP_SCRIPT
# grep -v "^#" ./scripts/train_v2_stage1.sh >> $TMP_SCRIPT
# grep -v "^#" ./scripts/train_v2_stage2.sh >> $TMP_SCRIPT

sbatch $TMP_SCRIPT

rm $TMP_SCRIPT
