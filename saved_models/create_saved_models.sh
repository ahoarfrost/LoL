#!/bin/bash

#SBATCH --partition=main                       # Partition (job queue)
#SBATCH --job-name=create                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=64000                                  # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.create_saved_models.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python create_saved_models.py --n_cpus 14
