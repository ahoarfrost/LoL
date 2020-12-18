#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=analysis                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.analysis_2M_preds.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


#analysis for metagenome 2M predictions
python analysis_preds.py