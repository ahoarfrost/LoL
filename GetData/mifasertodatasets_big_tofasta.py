#script to produce some clean train and valid csv files for annotation classifier
#ran this interactively
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import csv

train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/train/mifaser_train.csv')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/valid/mifaser_valid.csv')
trnout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_train.fasta')
valout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/mifaser_valid_candidates.fasta')

print('writing train')
train = pd.read_csv(train_path)
train_records = []
for ix,row in train.iterrows():
    if ix % 10000000 == 0:
        print('processing ix',ix,'out of',len(train),'rows')
    record = SeqRecord(Seq(row['seq']),id=row['annotation'],description=row['run']+'_'+str(ix)+'_'+row['annotation'])
    train_records.append(record)
with open(trnout,'a') as handle:
        SeqIO.write(train_records,handle,'fasta')

print('writing valid')
valid = pd.read_csv(valid_path)
valid_records = []
for ix,row in valid.iterrows():
    if ix % 10000000 == 0:
        print('processing ix',ix,'out of',len(valid),'rows')
    record = SeqRecord(Seq(row['seq']),id=row['annotation'],description=row['run']+'_'+str(ix)+'_'+row['annotation'])
    valid_records.append(record)
with open(valout,'a') as handle:
        SeqIO.write(valid_records,handle,'fasta')

