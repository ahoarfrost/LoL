#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=getgtdb          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.getGTDBrepgenomes_all.out    # STDOUT output file 

echo downloading bacterial genomes...
python GetGTDBrepGenomes.py

echo downloading archaeal genomes...
python GetGTDBrepGenomes_arc.py
