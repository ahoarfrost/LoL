#script to produce some clean train and valid csv files for annotation classifier
#ran this interactively
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import csv
from collections import Counter

train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/train/mifaser_train.csv')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/valid/mifaser_valid.csv')

train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)
train['anno3'] = ['.'.join(x.split('.')[0:3]) for x in train['annotation']]  
valid['anno3'] = ['.'.join(x.split('.')[0:3]) for x in valid['annotation']]  

def to_fasta(df,colname,outpath,subsample=True):
    annos = list(set(df[colname]))
    counts = Counter(df[colname]).most_common()
    c = [x[1] for x in counts]
    max_seqs = int(np.mean(c))
    for anno in annos:
        records = []
        fout = str(anno)+'.fasta'
        print('processing annotation',anno)
        subset = df[df[colname]==anno]
        if subsample and len(subset) > max_seqs:
            #randomly sample down train set to max_seqs
            subset = subset.sample(n=max_seqs)  
        for ix,row in subset.iterrows():
            record = SeqRecord(Seq(row['seq']),description=row['run']+'_'+str(ix)+'_'+row[colname])
            records.append(record)
        with open(outpath+fout,'a') as handle:
            SeqIO.write(records,handle,'fasta') 

print('writing train anno4')
outpath = '/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno4/train/'
to_fasta(train,'annotation',outpath=outpath)

print('writing valid anno4')
outpath = '/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno4/valid_candidates/'
to_fasta(valid,'annotation',outpath=outpath,subsample=False)

print('writing train anno3')
outpath = '/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/train/'
to_fasta(train,'anno3',outpath=outpath)

print('writing valid anno3')
outpath = '/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/data/anno3/valid_candidates/'
to_fasta(valid,'anno3',outpath=outpath,subsample=False)
