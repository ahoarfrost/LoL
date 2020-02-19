#!/bin/bash

#SBATCH --partition=p_ah1114_t                       # Partition (job queue)
#SBATCH --job-name=ksdroplong                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=128000                                  # Real memory (RAM) required (MB)
#SBATCH --time=14-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.searchlm_KSDrop_longcycle.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

nvidia-smi

#python -m torch.distributed.launch --nproc_per_node=2 searchlm_KSDrop.py
python searchlm_KSDrop_longcycle.py