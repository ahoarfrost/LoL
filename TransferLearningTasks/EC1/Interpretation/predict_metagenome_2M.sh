#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=pre72c                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.get_preds_ERR599072_2M.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


#get preds for ERR598962 ERR599055 ERR598957 ERR599072
python get_metagenome_preds_2M.py --max_seqs 2000000 --metagenome_id ERR599072 --n_cpus 28