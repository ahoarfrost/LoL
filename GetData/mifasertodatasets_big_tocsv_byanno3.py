#script to produce some clean train and valid csv files for annotation classifier
#ran this interactively 
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

def upsample_seqs(records,max_seqs):
    multiplier = int(max_seqs/len(records))
    new_records = multiplier*records
    return new_records

def create_csv(outfile, in_path, max_seqs):
    contents = [x for x in in_path.iterdir() if x.suffix=='.fasta']
    with open(outfile, 'w') as out_data:
        data_writer = csv.writer(out_data)
        data_writer.writerow(['run','seq','annotation'])
        #for each fasta...
        for fasta in contents:
        #for each of these reads, extract the run, seq, and annotation, and the labels from the filename
            records = []
            for record in SeqIO.parse(fasta,"fasta"): 
                run = str(record.description).split('_')[0].split(' ')[-1]
                anno = str(record.description).split('_')[2]
                seq = str(record.seq)
                record = [run, seq, anno]
                records.append(record)
            #if number of records is too low, upsample
            if max_seqs:
                records = upsample_seqs(records,max_seqs)
            #write rows with run, seq, and annotation label
            data_writer.writerows(records)
'''
print('processing anno4')
train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno4/train/')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno4/valid_filtered/')
trnout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/train/mifaser_train.csv')
valout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/valid/mifaser_valid.csv')

create_csv(outfile=trnout, in_path=train_path, max_seqs=52353) #this is the mean, median is 12674
create_csv(outfile=valout, in_path=valid_path, max_seqs=None)
'''
print('processing anno3')
train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/train/')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/valid_filtered/')
trnout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno3/train/mifaser_train.csv')
valout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno3/valid/mifaser_valid.csv')

create_csv(outfile=trnout, in_path=train_path, max_seqs=360533) #this is the mean, median is 87016
create_csv(outfile=valout, in_path=valid_path, max_seqs=None)

train = pd.read_csv(trnout)
train = train.sample(frac=1)
train.reset_index(drop=True)
train.to_csv(trnout,index=False)

valid = pd.read_csv(valout)
valid = valid.sample(frac=1)
valid.reset_index(drop=True)
valid.to_csv(valout,index=False)
