#!/bin/bash

#SBATCH --partition=gpu                       # Partition (job queue)
#SBATCH --gres=gpu:2
#SBATCH --constraint=pascal
#SBATCH --job-name=trnmet                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.train_metalbinding_clasall.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

python -m torch.distributed.launch --nproc_per_node=2 TrainMetalBinding.py --n_cpus 28

#python -m torch.distributed.launch --nproc_per_node=2 TrainMetalBinding_round0.py --n_cpus 28

#kill $(ps aux | grep "TrainMetalBinding" | grep -v grep | awk '{print $2}')

#python -m torch.distributed.launch --nproc_per_node=2 TrainMetalBinding_round1.py --n_cpus 28

#kill $(ps aux | grep "TrainMetalBinding" | grep -v grep | awk '{print $2}')

#python -m torch.distributed.launch --nproc_per_node=2 TrainMetalBinding_round2.py --n_cpus 28

#kill $(ps aux | grep "TrainMetalBinding" | grep -v grep | awk '{print $2}')

#python -m torch.distributed.launch --nproc_per_node=2 TrainMetalBinding_round3.py --n_cpus 28
