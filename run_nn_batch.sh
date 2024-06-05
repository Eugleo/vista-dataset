#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=single
#SBATCH --job-name=vlm_benchmark_models_evaluate
#SBATCH --output=logs/job_%j.log

config_file=$1

cd /data/evan_gunter/vlm-benchmark
source venv/bin/activate

vlm evaluate $config_file
