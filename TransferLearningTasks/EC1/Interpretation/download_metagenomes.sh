#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=dwn                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.cut_metagenomes.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


for srr in ERR598962 ERR599055 ERR598957 ERR599072;
do
brombergdump --split-spot --print-read-nr --outdir /scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/ $SRR
done