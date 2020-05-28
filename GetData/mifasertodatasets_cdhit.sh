#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=cleanmifaser          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.mifasertodatasets_cdhitvalid.out    # STDOUT output file 

python mifasertodatasets_cdhit.py

for fasta in `ls /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_train/`; do
    echo ------------>> $fasta 
    cd-hit-est-2d -i /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_train/$fasta -i2 /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_valid_candidates/$fasta -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_valid_filtered/$fasta -c 0.8 -d 0 -n 4 -M 110000 -T 28
done

python mifasertodatasets_cdhitvalid.py

while read f; do
    rm /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_train/$f
done < /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/train_to_remove.txt 