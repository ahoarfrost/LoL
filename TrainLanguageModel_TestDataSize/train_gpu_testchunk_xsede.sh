#!/bin/bash

#SBATCH --partition=GPU_AI                       # Partition (job queue)
#SBATCH --gres=gpu:volta16:8
#SBATCH --job-name=testchunk                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --time=2-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_GTDB_read_LM_testchunk.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

#I used to have rounds run in loop in the script, but python wasn't doing a good job of garbage collecting the databunch and got memory error
#running script separately n times avoids this, clears memory completely between scripts
for chunk in {0..9}
do
    echo training $chunk
    python -m torch.distributed.launch --nproc_per_node=8 train_GTDB_read_LM_testchunk_xsede.py --chunk $chunk --maxseq 1000 --n_cpus 28
done