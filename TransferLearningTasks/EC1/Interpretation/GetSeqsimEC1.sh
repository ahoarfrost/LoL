#!/bin/bash

#SBATCH --partition=main                       # Partition (job queue)
#SBATCH --job-name=ec1genesim                       # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000
#SBATCH --time=3-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.GetSeqsimEC1_genes.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


#get within-EC seqsim for EC1 groups in validation set
python GetSeqsimEC1_genes.py