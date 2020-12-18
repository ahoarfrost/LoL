#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=split          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=16000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.CreateGTDBrepCDScsv_train_parsedclass.out    # STDOUT output file 


echo processing valid
while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv/$line /scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv_valid_parsedclass/
done < /scratch/ah1114/LoL/data/validnew_parsedclass_filenames_csv.txt

echo processing train
while IFS= read -r line; do
    cp /scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv/$line /scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv_train_parsedclass/
done < /scratch/ah1114/LoL/data/trainnew_parsedclass_filenames_csv.txt

echo Done! 