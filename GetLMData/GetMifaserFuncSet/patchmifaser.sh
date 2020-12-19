#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=patch2          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.patchmifaser2.out    # STDOUT output file 

cd /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages

runs=(ERR1726943 ERR209626 ERR1135232 SRR1171647 SRR5091465)
downloadedtoreannotate=(ERR1726943 ERR209626 ERR1135232 SRR1171647 SRR5091465 ERR2200674 DRR046818 ERR2699809 SRR1574704 SRR830624 SRR5234512) 
numthread=28

#for run in ${runs[*]}
#do
#    brombergdump $run 
#done

#echo processing ERR1726943
#mifaser -l ERR1726943_1.fastq ERR1726943_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/$srr -d GS+ -m -t $numthread 
echo processing ERR209626
mifaser -l ERR209626_1.fastq ERR209626_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/ERR209626 -d GS+ -m -t $numthread 
echo processing ERR1135232
mifaser -l ERR1135232_1.fastq ERR1135232_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/ERR1135232 -d GS+ -m -t $numthread 
echo processing SRR1171647
mifaser -l SRR1171647_1.fastq SRR1171647_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR1171647 -d GS+ -m -t $numthread 
echo processing SRR5091465
mifaser -l SRR5091465_1.fastq SRR5091465_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR5091465 -d GS+ -m -t $numthread 
echo processing ERR2200674
mifaser -l ERR2200674_1.fastq ERR2200674_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/ERR2200674 -d GS+ -m -t $numthread 
echo processing DRR046818
mifaser -l DRR046818_1.fastq DRR046818_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/DRR046818 -d GS+ -m -t $numthread 
echo processing ERR2699809
mifaser -l ERR2699809_1.fastq ERR2699809_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/ERR2699809 -d GS+ -m -t $numthread 
echo processing SRR1574704
mifaser -l SRR1574704_1.fastq SRR1574704_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR1574704 -d GS+ -m -t $numthread 
echo processing SRR830624
mifaser -l SRR830624_1.fastq SRR830624_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR830624 -d GS+ -m -t $numthread 
echo processing SRR5234512
mifaser -l SRR5234512_1.fastq SRR5234512_2.fastq -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/SRR5234512 -d GS+ -m -t $numthread 
