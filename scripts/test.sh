#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=sgpu_long
#SBATCH --error=/home/s28zabed_hpc/NarrativeEMNLP/NarrativeFinalVersion/logs/error_output.log
#SBATCH --output=/home/s28zabed_hpc/NarrativeEMNLP/NarrativeFinalVersion/logs/server_output.log
#SBATCH --mail-type=ALL
#SBATCH --gpus=4
#SBATCH --cpus-per-task=124
#SBATCH --ntasks=1
#SBATCH --mem=128GB

export PYTHONPATH='/home/s28zabed_hpc/NarrativeEMNLP/NarrativeFinalVersion/Prompting/'
#source /home/s28zabed_hpc/anaconda3/etc/profile.d/conda.sh
#conda activate NarrativeEMNLP
source ../../../.venv/bin/activate

export MKL_SERVICE_FORCE_INTEL=1

python test.py