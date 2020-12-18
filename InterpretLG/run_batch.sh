#!/bin/bash

#SBATCH --partition=main                       # Partition (job queue)
#SBATCH --job-name=embval                          # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.GetEmbs_MifaserValid.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python GetEmbs_EvenEnvSubset.py --n_cpus 28
