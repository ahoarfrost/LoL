#!/bin/bash

#SBATCH --partition=bromberg_1             # Partition (job queue)
#SBATCH --job-name=comparem         # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=40          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=192000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.GTDBrepGenomes_comparemagain.out    # STDOUT output file 

#make sure you're in the right pyenv virtualenv!
#comparem aai_wf --cpus 28 /scratch/ah1114/LoL/data/tinyGenomes /scratch/ah1114/LoL/data/tinyGenomes_CompareMout_oldcomparem
comparem similarity --cpus 40 --tmp_dir /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout_again/ /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout/genes/ /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout/genes/ /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout_again/similarity
#comparem aai --cpus 40 /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout/comparemagain/similarity/query_genes.faa /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout/comparemagain/similarity/hits_sorted.tsv /scratch/ah1114/LoL/data/GTDBrepGenomes_CompareMout/comparemagain/aai
