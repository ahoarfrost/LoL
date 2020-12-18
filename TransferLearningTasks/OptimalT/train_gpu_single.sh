#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|volta"
#SBATCH --job-name=trnTcat2                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.TrainOptimalT_clas2.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python TrainOptimalT_clas2.py --n_cpus 28
