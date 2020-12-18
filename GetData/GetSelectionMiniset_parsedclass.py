from pathlib import Path
from Bio import SeqIO
import numpy as np

train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train_parsedclass/')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid_parsedclass/')
out_train = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset_parsedclass/train/GTDB_parsedclass100_train.fna')
out_valid = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset_parsedclass/valid/GTDB_parsedclass100_valid.fna')

train_files = [x for x in train_path.iterdir()]
val_files = [x for x in valid_path.iterdir()]
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
            if row >= 100:
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
            if row >= 100:
                break
        SeqIO.write(records, valhandle, "fasta")
valhandle.close()