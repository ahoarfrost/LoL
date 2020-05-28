#script to take the sequences out of the chunked PDB_to_ENA DNA sequences and compile them into a dataframe
#I ran this in interactive mode 

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

root = Path('/scratch/ah1114/LoL/data/')
files = [x for x in (root/'PDB_to_ENA_chunked').resolve().iterdir()]
fout = '/scratch/ah1114/LoL/data/PDBtoENA_clean.csv'

ids = pd.read_csv('/home/ah1114/LanguageOfLife/Interpret/Bac_UniprotToENA.tab',sep='\t')
ids.columns = ['PDBid','ENAid']

seqs = pd.DataFrame()  
toremove = []
for fasta in files:
    ena = fasta.stem
    #for each of these files, extract the reads, and the ENA id from the filename
    records = []
    for record in SeqIO.parse(fasta,"fasta"): 
        seq = str(record.seq)
        record = [ena,seq]
        records.append(record)
    if len(records)==0:
        print('no records, skipping',fasta)
        toremove.append(fasta)
    else:
        #append records to seqs df
        seqs = seqs.append(records)

seqs.columns = ['ENAid','seq']

ids.index = ids['ENAid']
joined = seqs.join(ids[['PDBid']],on='ENAid')
joined.to_csv(fout,index=False)