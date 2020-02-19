#script to take the read_map output of mifaser and produce some clean train and valid csv files for annotation classifier
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

runs = pd.read_csv("/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/TrainSmall100_EvenEnv_RunIDS.csv", header=None)
runs = list(runs[0])
runs.remove('DRR046818')  #removing this one that's still working
root = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/')
trnout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/mifaser_train.csv')
valout = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/mifaser_valid.csv')

with open(trnout, 'w') as out_train:
    with open(valout, 'w') as out_valid:
        train_writer = csv.writer(out_train)
        valid_writer = csv.writer(out_valid)
        train_writer.writerow(['run','seq','annotation'])
        valid_writer.writerow(['run','seq','annotation'])
        for run in runs:
            print(run,'...')
            #get a list of all the files in read_map for this run 
            contents = [x for x in Path(root/run/'read_map').resolve().iterdir()]
            #for each of these files, extract the reads, and the labels from the filename
            for fasta in contents:
                train_records = []
                valid_records = []
                records = []
                label = fasta.stem
                for record in SeqIO.parse(fasta,"fasta"): 
                    seq = str(record.seq)
                    record = [run, seq, label]
                    records.append(record)
                #split records into train/valid 80/20
                trn, val = np.split(records, [int(len(records)*0.8)]) 
                train_records.extend([list(x) for x in trn])
                valid_records.extend([list(x) for x in val])
                #write a row with run, seq, and annotation label
                train_writer.writerows(train_records)
                valid_writer.writerows(valid_records)

    