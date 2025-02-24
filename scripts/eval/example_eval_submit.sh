#!/bin/bash
#SBATCH --account vjgo8416-arc-extract
#SBATCH --qos turing
#SBATCH --job-name dummy-eval-job
#SBATCH --time 0-5:0:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --output ./slurm_logs/dummy-eval-job-%j.out

# Load required modules here
module purge
module load baskerville

# Activate venv
source /bask/projects/v/vjgo8416-arc-extract/ARC-Event-Extraction/.venv/bin/activate

# Run
python scripts/eval/inference.py --config_path scripts/configs/example_job.yaml
