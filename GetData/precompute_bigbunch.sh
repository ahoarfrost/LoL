#!/bin/bash

#SBATCH --partition=mem             # Partition (job queue)
#SBATCH --job-name=bigbover          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=40          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=1000000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.precompute_bigbunch_overcommit.out    # STDOUT output file 

echo 1 > /proc/sys/vm/overcommit_memory
python precompute_bigbunch.py
