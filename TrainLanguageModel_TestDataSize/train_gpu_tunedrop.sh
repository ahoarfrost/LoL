#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --job-name=tunedrop                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_GTDB_read_LM_tunedrop_%a.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env
#SBATCH --array=1-4

dropmults=("dummy" 0.01 0.05 0.1 0.2)
dm=${dropmults[${SLURM_ARRAY_TASK_ID}]}

echo training with drop_mult $dm
python -m torch.distributed.launch --nproc_per_node=2 train_GTDB_read_LM_testclass_tunedropmult.py --drop_mult $dm --n_cpus 28
