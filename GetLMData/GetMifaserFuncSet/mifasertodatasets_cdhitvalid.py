import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

valid_cand_outpath = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_valid_candidates')
valid_outpath = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_valid')

#for any valid with too few inputs (fewer than 180), remove train and valid example; 
#otherwise, take first 180 seqs and include in final valid set
contents = [x for x in valid_cand_outpath.iterdir()]
for annotation in contents:
    print('processing ec number',annotation)
    valid_records = []
    label = annotation.stem
    for ix,record in enumerate(SeqIO.parse(annotation,"fasta")):
        if ix < 180:
            valid_records.append(record)

    if len(valid_records) >= 180:
        with open(valid_outpath/Path(label+'.fasta'),'a') as handle:
            SeqIO.write(valid_records,handle,'fasta')
    else:
        #record train to remove
        with open('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/train_to_remove.txt','a') as handle:
            handle.write(label+'.fasta\n')