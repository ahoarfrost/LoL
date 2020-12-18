#!/bin/bash

#SBATCH --partition=main            # Partition (job queue)
#SBATCH --job-name=simphylum          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000                   # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export you current env to the job env
#SBATCH --output=slurm.GetSeqEmbHomData_phylum.out    # STDOUT output file 

python GetSeqEmbHomData.py --fin /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_HomEmb_out/OG_SeqEmbs_phylum.pkl --fout /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_Similarities_out/Homolog_Emb_Seq_Comp_phylum.csv
