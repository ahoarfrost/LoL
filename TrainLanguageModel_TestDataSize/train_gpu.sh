#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --job-name=testchunk                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_GTDB_read_LM_testchunk.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

#I used to have rounds run in loop in the script, but python wasn't doing a good job of garbage collecting the databunch and got memory error
#running script separately n times avoids this, clears memory completely between scripts
#maxseq of 10000 fails; 
for chunk in {0..5}
do
    echo training $chunk
    python -m torch.distributed.launch --nproc_per_node=2 train_GTDB_read_LM_testchunk.py --chunk $chunk --maxseq 3000 --n_cpus 28
done