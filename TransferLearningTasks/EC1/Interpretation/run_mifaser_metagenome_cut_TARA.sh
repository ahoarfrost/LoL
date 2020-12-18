#!/bin/bash

#SBATCH --partition=main                       # Partition (job queue)
#SBATCH --job-name=annocut5                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.run_mifaser_metagenome_cut_TARA5.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env

numthread=28

#get mifaser annotations for ERR598981 ERR599063 ERR599115 ERR599052 ERR599020 ERR599039 ERR599048 ERR599105 ERR599125 ERR599176 ERR599076 ERR598989 ERR598964 ERR598963 ERR3589593 ERR3589586
for srr in ERR598981 ERR599063 ERR599115 ERR599052 ERR599020 ERR599039 ERR599048 ERR599105 ERR599125 ERR599176 ERR599076 ERR598989 ERR598964 ERR598963 ERR3589593 ERR3589586;
do 
    if [ ! -d /scratch/ah1114/LoL/TransferLearningTasks/EC1/mifaser_metagenomes/$srr'_cut20M' ]; then #if no folder for this run exists (mifaser hasn't started on it yet), start processing
        echo processing $srr cut...
        start=`date +%s`
        mifaser -f /scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/$srr'_cut20M.fastq' -o /scratch/ah1114/LoL/TransferLearningTasks/EC1/mifaser_metagenomes/$srr'_cut20M' -d GS+ -m -t $numthread -q
        end=`date +%s`
        runtime=$((end-start))
        echo runtime is $runtime
    else
        echo "mifaser out folder $srr _cut exists, skipping"
    fi
done 