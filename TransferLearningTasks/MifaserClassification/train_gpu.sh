#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --job-name=trnm2                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_mifaser_clas2.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


python -m torch.distributed.launch --nproc_per_node=2 TrainMifaserClas2.py --n_cpus 28
