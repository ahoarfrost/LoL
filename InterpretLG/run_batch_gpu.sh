#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|volta"
#SBATCH --job-name=embvlgpu                          # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.GetEmbs_MifaserValid_gpu.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python GetEmbs_MifaserValid.py --n_cpus 28
