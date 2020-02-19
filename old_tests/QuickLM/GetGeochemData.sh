#!/bin/bash

#SBATCH --partition=gpu             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=$1          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --mem=4000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --gres=gpu:1                # select 1 gpu
#SBATCH --constraint=pascal  

module purge 

srun python GetGeochemData.py $1
