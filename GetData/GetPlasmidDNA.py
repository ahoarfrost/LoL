import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

root = Path('/scratch/ah1114/LoL/data/')
#files = list(pd.read_csv((root/'trainnew_parsedorder_filenames.txt').resolve(),sep='\t',header=None)[0]) 
#files = list(pd.read_csv((root/'trainnewparsed_filenames.txt').resolve(),sep='\t',header=None)[0]) 
files = list(pd.read_csv((root/'validnew_parsedgenus_filenames.txt').resolve(),sep='\t',header=None)[0]) 
files = ['.'.join(x.split('.')[:-1])+'_genomic.fna' for x in files]

plasmids = []
records = []
lens = []
for fasta in files:
    #for each of these files, extract the reads, 
    fpath = (root/'GTDBrepGenomes'/fasta).resolve()
    for record in SeqIO.parse(fpath,"fasta"): 
        description = record.description
        if 'plasmid' in description:
            plasmids.append(fpath)
            lens.append(len(record.seq))

#in orderparsed there were 56 genomes with plasmids, 4.52M bases, ~33.5k inputs
#in familyparsed there were 60 genomes with plasmids, 6885834 bases, ~51k inputs
#in genusparsed there are 378 genomes with plasmids, so 2 or 3 hundred thousand inputs-ish (x2 for both labels)
#in genusparsed validset there are found 64 genomes plasmids, 6684610 total nucleotides, ~50k inputs (x2 for both labels)
print('found',len(plasmids),'plasmids')
print('total plasmid sequence length of',np.sum(lens),'nucleotides')

with open('genomes_with_plasmids_genusparsed_valid.txt', 'w') as f:
    for item in plasmids:
        f.write("%s\n" % item)
