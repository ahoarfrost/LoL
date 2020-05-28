#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=anno3          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=28          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=anno3/slurm.mifasertodatasets_cdhit_byanno_anno3_%a.out    # STDOUT output file 
#SBATCH --array=1-185

#takes each annotation in mifaser reads and makes separate cdhit database for each anno (with max_seqs for train)
#echo making train and valid_candidates fasta databases...
#python mifasertodatasets_big_tofasta_byanno.py

#run cdhit for each fasta in anno3
fasta=$(ls /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/train/*.fasta | sed -n ${SLURM_ARRAY_TASK_ID}p)
cd-hit-est-2d -i /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/train/$fasta -i2 /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/valid_candidates/$fasta -o /scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/valid_filtered/$fasta -c 0.8 -d 0 -n 4 -M 110000 -T 28

#oversample rarer train sets and write to csv
#python mifasertodatasets_cdhitvalid.py
