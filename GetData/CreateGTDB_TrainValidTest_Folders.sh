#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=split          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=16000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.split_trainvalidtestGTDB.out    # STDOUT output file 

while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/$line /scratch/ah1114/LoL/data/GTDB_chunked_train/
done < /scratch/ah1114/LoL/data/train_filenames.txt

while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/$line /scratch/ah1114/LoL/data/GTDB_chunked_valid/
done < /scratch/ah1114/LoL/data/valid_filenames.txt

while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/$line /scratch/ah1114/LoL/data/GTDB_chunked_test/
done < /scratch/ah1114/LoL/data/test_filenames.txt