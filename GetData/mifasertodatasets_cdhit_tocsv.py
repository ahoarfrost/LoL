#script to produce some clean train and valid csv files for annotation classifier
#ran this interactively
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data/train/')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data/valid/')
trnout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data/mifaser_train.csv')
valout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data/mifaser_valid.csv')

def create_csv(outfile, in_path):
    contents = [x for x in in_path.iterdir()]
    with open(outfile, 'w') as out_data:
        data_writer = csv.writer(out_data)
        data_writer.writerow(['seq','annotation'])
        for ix,ec in enumerate(contents):
            if ix % 100 == 0:
                print('processing ec',ix,'out of',len(contents),':',ec.stem,'...')
            #for each of these files, extract the reads, and the labels from the filename
            records = []
            label = ec.stem
            for record in SeqIO.parse(ec,"fasta"): 
                seq = str(record.seq)
                record = [seq, label]
                records.append(record)
            #write a row with run, seq, and annotation label
            data_writer.writerows(records)

create_csv(outfile=trnout, in_path=train_path)
create_csv(outfile=valout, in_path=valid_path)

    