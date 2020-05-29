#!/bin/bash

#SBATCH --partition=bromberg_1                       # Partition (job queue)
#SBATCH --job-name=dwnENA                           # Assign an short name to your job
#SBATCH --nodes=1                                   # Number of nodes you require
#SBATCH --mem=16000                                  # Real memory (RAM) required (MB)
#SBATCH --time=1-00:00:00                           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.Download_PDB_to_ENA.out     # STDOUT output file
#SBATCH --export=ALL                                # Export you current env to the job env


#download the DNA CDS sequences for PDB structures I ran topmatch on
#note the CDS sequence is for the actual part of the gene where the structural alignment is done, so I chose to do this rather than the whole gene, more apples to apples 
#I used the online ENA mapper at https://www.uniprot.org/uploadlists/ to get EMBL CDS IDs from Uniprot - these are saved in Bac_UniprotToENA.tab and EColi_UniprotToENA.tab
#The Uniprot IDs were collected from the online interface of PDB; I got 1-chain proteins at 50% identity cutoff below 2A resolution for all Bacteria = 293 proteins - these are stored in PDBForTopmatch_1chain50id2ABacteria.csv (also same for EColi); and I have pairwise topmatch structural alignments for all of these
#note a few Uniprot IDs didn't have EMBL matches, these are noted in the .not.tab files; it's only a few so I will ignore
#also note there are multiple EMBL sequences for one Uniprot often

#CDS sequences can be downloaded directly from the EMBL api with the url:
#https://www.ebi.ac.uk/ena/browser/api/fasta/$EMBLID?download=true where $EMBLID is the ENA ID such as AAA23651.1 (see Bac_UniprotToENA.tab)

tail -n +2 /home/ah1114/LanguageOfLife/Interpret/Bac_UniprotToENA_todownload.tab | while read p; do
    curl -o /scratch/ah1114/LoL/data/PDB_to_ENA_seqs/$p.fasta https://www.ebi.ac.uk/ena/browser/api/fasta/$p?download=true
done 