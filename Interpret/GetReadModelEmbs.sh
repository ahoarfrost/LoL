#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=getembs3                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.GetReadModelEmbs_singlereads3.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python GetReadModelEmbs_singlereads.py 3 