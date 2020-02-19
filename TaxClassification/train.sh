#!/bin/bash

#SBATCH --partition=p_ah1114_t                       # Partition (job queue)
#SBATCH --job-name=trnclas                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=14-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.trainTaxClasGTDB_k1_1000seq.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python trainTaxClasGTDB_k1_1000seq.py