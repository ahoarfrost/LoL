#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=miniset          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=12          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=64000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.GetSelectionMiniset.out    # STDOUT output file 

python GetSelectionMiniset.py
