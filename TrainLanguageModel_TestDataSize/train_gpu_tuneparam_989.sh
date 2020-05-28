#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --job-name=tune989                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_GTDB_read_LM_tuneclass_989_%a.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env
#SBATCH --array=2,1

#I used to have rounds run in loop in the script, but python wasn't doing a good job of garbage collecting the databunch and got memory error
#running script separately n times avoids this, clears memory completely between scripts
echo training with wd 1e-${SLURM_ARRAY_TASK_ID} and moms 0.98 0.9
python -m torch.distributed.launch --nproc_per_node=2 train_GTDB_read_LM_testclass_tuneparam.py --wd 1e-${SLURM_ARRAY_TASK_ID} --moms 0.98 0.9 --n_cpus 28
