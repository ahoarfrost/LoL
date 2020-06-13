#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=predval                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.get_preds_validset.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


#get preds for EC1 validation set
#can use these to look at prob distributions
python get_validset_preds.py --n_cpus 28