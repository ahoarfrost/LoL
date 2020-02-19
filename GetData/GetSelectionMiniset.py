from pathlib import Path
from Bio import SeqIO
import numpy as np

data_path = Path('/scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/')
out_train = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset/train/GTDBrepGenomes10_train.fna')
out_valid = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset/valid/GTDBrepGenomes10_valid.fna')

files = [x for x in data_path.iterdir()]
train_files, val_files = np.split(files, [int(len(files)*0.8)]) 
print(len(train_files),len(val_files))

print('processing training files')
with open(out_train, "a") as handle:
    for fasta in train_files:
        print(fasta)
        row = 0
        records = []
        for record in SeqIO.parse(fasta, "fasta"):
            records.append(record)
            row += 1
            if row >= 10:
                break
        SeqIO.write(records, handle, "fasta")
handle.close()

print('processing valid files')
with open(out_valid, "a") as valhandle:
    for fasta in val_files:
        print(fasta)
        row = 0
        records = []
        for record in SeqIO.parse(fasta, "fasta"):
            records.append(record)
            row += 1
            if row >= 10:
                break
        SeqIO.write(records, valhandle, "fasta")
valhandle.close()