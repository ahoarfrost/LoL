#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --job-name=lmdatatest                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_GTDB_read_LM_datatest_%a.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env
#SBATCH --array=500,1000

#I used to have rounds run in loop in the script, but python wasn't doing a good job of garbage collecting the databunch and got memory error
#running script separately n times avoids this, clears memory completely between scripts
skiprows=`shuf -i 0-10000 -n 1`
echo training with maxseqs ${SLURM_ARRAY_TASK_ID} and skiprows $skiprows
python -m torch.distributed.launch --nproc_per_node=2 train_GTDB_read_LM_datatest.py --maxseq ${SLURM_ARRAY_TASK_ID} --skiprows $skiprows --n_cpus 28
