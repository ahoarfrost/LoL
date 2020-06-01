#!/bin/bash

#SBATCH --partition=bromberg_1            # Partition (job queue)
#SBATCH --job-name=homemb          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.GetReadModelEmbs_singlereads_Homologs_class.out    # STDOUT output file 

python GetReadModelEmbs_singlereads_Homologs_class.py
