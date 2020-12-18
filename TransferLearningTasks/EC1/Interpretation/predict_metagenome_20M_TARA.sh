#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|volta"
#SBATCH --job-name=pre586                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=14
#SBATCH --mem=64000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.get_preds_ERR3589586_20M_TARA.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


#get preds for ERR598981 ERR599063 ERR599115 ERR599052 ERR599020 ERR599039 ERR599048 ERR599105 ERR599125 ERR599176 ERR599076 ERR598989 ERR598964 ERR598963 ERR3589593 ERR3586717;
python get_metagenome_preds_20M_TARA.py --max_seqs 20000000 --metagenome_id ERR3589586 --n_cpus 14