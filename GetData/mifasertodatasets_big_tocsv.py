#script to produce some clean train and valid csv files for annotation classifier
#ran this interactively 
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_train.fasta')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_valid.fasta')
trnout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed/train/mifaser_train.csv')
valout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed/valid/mifaser_valid.csv')

def create_csv(outfile, in_path):
    with open(outfile, 'w') as out_data:
        data_writer = csv.writer(out_data)
        data_writer.writerow(['run','seq','annotation'])
        #for each of these reads, extract the run, seq, and annotation, and the labels from the filename
        records = []
        for record in SeqIO.parse(in_path,"fasta"): 
            run = str(record.description).split('_')[0].split(' ')[-1]
            anno = str(record.description).split('_')[2]
            seq = str(record.seq)
            record = [run, seq, anno]
            records.append(record)
        #write rows with run, seq, and annotation label
        data_writer.writerows(records)

create_csv(outfile=trnout, in_path=train_path)
create_csv(outfile=valout, in_path=valid_path)

    