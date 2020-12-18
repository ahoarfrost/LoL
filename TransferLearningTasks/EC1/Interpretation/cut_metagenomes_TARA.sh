#!/bin/bash

#SBATCH --partition=main                       # Partition (job queue)
#SBATCH --job-name=cut                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.cut_metagenomes_TARA2.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

cd /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/

#for srr in ERR598981 ERR599063 ERR599115 ERR599052 ERR599020 ERR599039 ERR599048 ERR599105 ERR599125 ERR599176 ERR599076 ERR598989 ERR598964 ERR598963 ERR3589593 ERR3589586;
for srr in ERR3589586;
do
    #download with fasterq-dump or brombergdump if haven't run download_metagenomes_TARA.sh
    brombergdump --split-spot --print-read-nr --outdir /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/ $srr
    #fasterq-dump $srr -O /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes 

    echo processing $srr ...
    python /home/ah1114/LanguageOfLife/TransferLearningTasks/EC1/Interpretation/cut_metagenome_TARA.py --metagenome_id $srr --max_seqs 20000000

    #delete original file to save space
    rm /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/$srr.fastq
done