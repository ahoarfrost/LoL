#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=patch4          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.patchmifaser4.out    # STDOUT output file 

cd /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages

runs=(ERR1726943 ERR209626 ERR1135232 SRR1171647 SRR5091465)
downloadedtoreannotate=(ERR1726943 ERR209626 ERR1135232 SRR1171647 SRR5091465 ERR2200674 DRR046818 ERR2699809 SRR1574704 SRR830624 SRR5234512) 
numthread=28

#for run in ${runs[*]}
#do
#    brombergdump $run 
#done

echo processing DRR046818
mifaser -l DRR046818_1.fastq DRR046818_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/DRR046818 -d GS+ -m -t $numthread 
