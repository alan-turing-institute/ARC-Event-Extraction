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

# Set Ollama Port

export OLLAMA_PORT=$[($RANDOM % 1000) + 20000]
echo "port is ${OLLAMA_PORT}"

# Set enviroment up for Ollama
export PATH=${PATH}:/bask/projects/v/vjgo8416-arc-extract/ollama_install/usr/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/bask/projects/v/vjgo8416-arc-extract/ollama_install/usr/lib
export OLLAMA_MODELS=/bask/projects/v/vjgo8416-arc-extract/ollama_install/models
export OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT}

echo "host is ${OLLAMA_HOST}"

echo "Starting Ollama Server"

ollama serve &
echo "Waiting for server to load up"
sleep 5

echo "About to run script"

# Run
python scripts/eval/inference.py --config_path scripts/configs/test/bask_test_ed.yaml --ollama_port=${OLLAMA_PORT}
