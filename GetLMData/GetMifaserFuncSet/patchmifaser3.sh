#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=patch3          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.patchmifaser3.out    # STDOUT output file 

cd /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages

downloadedtoreannotate3=(ERR1726943 ERR2699809 SRR1574704 SRR830624 SRR5234512) 
numthread=28

echo processing ERR1726943
mifaser -l ERR1726943_1.fastq ERR1726943_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/ERR1726943 -d GS+ -m -t $numthread 
echo processing ERR2699809
mifaser -l ERR2699809_1.fastq ERR2699809_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/ERR2699809 -d GS+ -m -t $numthread 
echo processing SRR1574704
mifaser -l SRR1574704_1.fastq SRR1574704_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR1574704 -d GS+ -m -t $numthread 
echo processing SRR830624
mifaser -l SRR830624_1.fastq SRR830624_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR830624 -d GS+ -m -t $numthread 
echo processing SRR5234512
mifaser -l SRR5234512_1.fastq SRR5234512_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR5234512 -d GS+ -m -t $numthread 
