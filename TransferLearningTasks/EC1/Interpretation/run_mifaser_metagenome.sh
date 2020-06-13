#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=anno                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.run_mifaser_metagenome.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

numthread=28

#get preds for ERR598962 ERR599055 ERR598957 ERR599072
for srr in ERR598962 ERR599055 ERR598957 ERR599072
do 
    if [ ! -d "/scratch/ah1114/LoL/TransferLearningTasks/EC1/mifaser_metagenomes/$srr" ]; then #if no folder for this run exists (mifaser hasn't started on it yet), start processing
        echo processing $srr ...
        start=`date +%s`
        mifaser -f /scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/$srr.fastq -o /scratch/ah1114/LoL/TransferLearningTasks/EC1/mifaser_metagenomes/$srr -d GS+ -m -t $numthread -q
        end=`date +%s`
        runtime=$((end-start))
        echo runtime is $runtime
    else
        echo "mifaser out folder $srr exists, skipping"
    fi
done