#!/bin/bash

#SBATCH --partition=mem             # Partition (job queue)
#SBATCH --job-name=comp0          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=40          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=1000000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.precompute_itemlist0.out    # STDOUT output file 

python precompute_itemlist.py 0 16000 0
#python precompute_itemlist.py 1 16000 16000
