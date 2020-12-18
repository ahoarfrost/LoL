#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=getgtdb          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.getGTDBrepgenomes_all_parsedclass.out    # STDOUT output file 

echo downloading class parsed genomes...
python GetGTDBrepGenomes_all_parsedclass.py

#patch the few archaeal genomes that are only SRA accessions
echo patching SRR accessions...
brombergdump SRR10597681 -O /scratch/ah1114/LoL/data/GTDBrepGenomes
brombergdump SRR10597694 -O /scratch/ah1114/LoL/data/GTDBrepGenomes
brombergdump SRR10597634 -O /scratch/ah1114/LoL/data/GTDBrepGenomes
brombergdump SRR10597626 -O /scratch/ah1114/LoL/data/GTDBrepGenomes