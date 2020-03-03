#!/bin/bash

#SBATCH --partition=mem             # Partition (job queue)
#SBATCH --job-name=diamondmem         # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=56          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=1500000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.computediamond_mem.out    # STDOUT output file 

diamond blastp -p 56 -q /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout_again/similarity/query_genes.faa -d /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout_again/similarity/query_genes.dmnd -e 1e-3 --id 30.0 --query-cover 70.0 -k 234580 -o /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout_again/similarity/comparem_hits_memdout -f 6 -t /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout_again/similarity -c 1 -b 8