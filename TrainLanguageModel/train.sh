#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=testlim                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=5-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.GTDBLMdatabunch_testlimits.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python save_GTDBLMdatabunch_valset.py