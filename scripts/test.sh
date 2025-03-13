#!/bin/bash
#SBATCH --time=00:50:00
#SBATCH --partition=A40short
#SBATCH --error=/home/s28zabed/simulation/logs/error_output.log
#SBATCH --output=/home/s28zabed/simulation/logs/server_output.log

source ../venv/bin/activate

python simulation.py
