#!/bin/bash

#SBATCH --partition=bromberg_1             # Partition (job queue)
#SBATCH --job-name=compress          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.compress_GTDBdata.out    # STDOUT output file 

cd /scratch/ah1114/LoL/data
tar -czf LMdata.tar.gz GTDB_chunked_train/ GTDB_chunked_valid/ GTDB_chunked_test/ GTDBdatabunch_validonly.pkl
