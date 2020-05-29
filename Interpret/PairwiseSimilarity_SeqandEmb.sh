#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=pairs                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --cpus-per-task=28
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=2-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.PairwiseSimilarity_SeqandEmb_withalignlength%a.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env
#SBATCH --array=0-7

echo processing chunk ${SLURM_ARRAY_TASK_ID} 

python PairwiseSimilarity_SeqandEmb_withalignlength.py ${SLURM_ARRAY_TASK_ID} 