#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=spltchunk          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=16000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.CreateGTDB_ChunkedFolders_parsedclass.out    # STDOUT output file 

echo processing valid
while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/$line /scratch/ah1114/LoL/data/GTDB_chunked_valid_parsedclass/
done < /scratch/ah1114/LoL/data/validnew_parsedclass_filenames.txt

echo processing train
while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/$line /scratch/ah1114/LoL/data/GTDB_chunked_train_parsedclass/
done < /scratch/ah1114/LoL/data/trainnew_parsedclass_filenames.txt

echo processing test
while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/$line /scratch/ah1114/LoL/data/GTDB_chunked_test_parsedclass/
done < /scratch/ah1114/LoL/data/testnew_parsedclass_filenames.txt

echo Done!