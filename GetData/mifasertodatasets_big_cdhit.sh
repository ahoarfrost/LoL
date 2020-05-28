#!/bin/bash

#SBATCH --partition=mem             # Partition (job queue)
#SBATCH --job-name=mifbig3          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks-per-node=40                  # total number of tasks across all nodes
#SBATCH --mem=1000000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.mifasertodatasets_big_cdhit3.out    # STDOUT output file 

#echo converting to fasta...
#python mifasertodatasets_big_tofasta.py

echo running cdhit...
cd-hit-est-2d -i /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_train.fasta -i2 /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_valid_candidates.fasta -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_valid3.fasta -c 0.8 -d 0 -n 4 -M 1000000 -T 40

echo converting to csv...
python mifasertodatasets_big_tocsv.py
