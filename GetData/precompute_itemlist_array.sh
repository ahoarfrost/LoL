#!/bin/bash

#SBATCH --partition=mem             # Partition (job queue)
#SBATCH --job-name=ilist          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=40          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=1000000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.precompute_trainparsed_itemlist_array%a.out    # STDOUT output file 
#SBATCH --array=0-3

echo processing chunk ${SLURM_ARRAY_TASK_ID} 

python precompute_itemlist_array.py ${SLURM_ARRAY_TASK_ID}

