#!/bin/bash

#SBATCH --partition=gpu             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=quickLM          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --mem=32000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.quickLM_morex2.out     # STDOUT output file
#SBATCH --error=slurm.quickLM_morex2.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --gres=gpu:2                 # select 2 gpu
#SBATCH --constraint=pascal   

module purge 

python TrainQuickLM.py 
