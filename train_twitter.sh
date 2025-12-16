#!/bin/bash
#SBATCH --job-name=mentalbert_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL


module purge
module load python/3.10
module load cuda/11.7

mkdir -p logs

source ~/envs/mentalbert/bin/activate

export HF_TOKEN="my token"

python twitter.py