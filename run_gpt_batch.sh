#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --partition=single
#SBATCH --job-name=vlm_benchmark_gpt
#SBATCH --output=logs/job_%j.log

config_file=$1

check_valid_paths=true
if [ ! -z "$2" ]
then
    if [ "$2" == "--no-validate" ]
    then
        check_valid_paths=false
    else
        echo "Ignoring second argument: $2"
    fi
fi

cd /data/evan_gunter/vlm-benchmark
source venv/bin/activate

if [ "$check_valid_paths" = true ]
then
    python /data/datasets/vlm_benchmark/tasks/real_life/validate_paths.py
fi

vlm evaluate $config_file
