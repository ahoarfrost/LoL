#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=getembs                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=192000                                  # Real memory (RAM) required (MB)
#SBATCH --time=14-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.GetReadModelEmbs.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python GetReadModelEmbs.py