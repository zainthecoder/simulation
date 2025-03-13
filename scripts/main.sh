#!/bin/bash
#SBATCH --time=00:50:00
#SBATCH --partition=sgpu_devel
#SBATCH --error=/home/s28zabed_hpc/simulation/logs/error_output.log
#SBATCH --output=/home/s28zabed_hpc/simulation/logs/server_output.log

source ../../.venv/bin/activate

python simulation.py
