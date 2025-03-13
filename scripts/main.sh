#!/bin/bash
#SBATCH --time=00:50:00
#SBATCH --partition=mlgpu_devel
#SBATCH --error=/home/s28zabed_hpc/simulation/logs/error_output.log
#SBATCH --output=/home/s28zabed_hpc/simulation/logs/server_output.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=124
#SBATCH --ntasks=1
#SBATCH --mem=128GB

source ../../.venv/bin/activate

python simulation.py
