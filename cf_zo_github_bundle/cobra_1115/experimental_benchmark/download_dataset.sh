#!/bin/bash
#SBATCH --job-name=longbench-dl
#SBATCH --account=MST114205
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=download_dataset.%j.out
#SBATCH --error=download_dataset.%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/work/hsuan1007/cobra_1115/experimental_benchmark

set -euo pipefail

# Uncomment and set your environment if needed.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <env-name>

python download_dataset.py
