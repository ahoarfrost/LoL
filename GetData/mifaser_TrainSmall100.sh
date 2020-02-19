#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=anno11          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.annotate11_Small100.out    # STDOUT output file 

cd /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages

runs=$(cat /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/TrainSmall100_EvenEnv_RunIDS.csv)
numthread=28

for srr in $runs
do 
    if [ ! -d "/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/$srr" ]; then #if no folder for this run exists (mifaser hasn't started on it yet), start processing
        echo processing $srr ...
        start=`date +%s`
        mifaser -f sra:$srr -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/$srr -d GS+ -m -t $numthread -q
        end=`date +%s`
        runtime=$((end-start))
        echo runtime is $runtime
    else
        echo "mifaser out folder $srr exists, skipping"
    fi
done