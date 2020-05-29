#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal
#SBATCH --job-name=trnmif4                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.TrainMifaserClas_anno4_round2.5.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python TrainMifaserClas_anno4_round2.py --n_cpus 28
