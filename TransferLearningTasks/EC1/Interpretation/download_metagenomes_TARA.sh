#!/bin/bash

#SBATCH --partition=main                       # Partition (job queue)
#SBATCH --job-name=dwn                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.download_metagenomes_TARA.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

cd /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/

for srr in ERR598981 ERR599063 ERR599115 ERR599052 ERR599020 ERR599039 ERR599048 ERR599105 ERR599125 ERR599176 ERR599076 ERR598989 ERR598964 ERR598963 ERR3589593 ERR3589586;
do
brombergdump --split-spot --print-read-nr --outdir /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/ $srr
done