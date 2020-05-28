#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal
#SBATCH --job-name=trainload                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_GTDB_read_LM_loadall.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python train_GTDB_read_LM_single_datatest.py 1
