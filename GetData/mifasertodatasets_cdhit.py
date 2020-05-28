#script to take the read_map output of mifaser and produce some clean train and valid csv files for annotation classifier
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import csv

runs = pd.read_csv("/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/TrainSmall100_EvenEnv_RunIDS.csv", header=None)
runs = runs[0].tolist()
root = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/')
outpath = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_all')
train_outpath = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_train')
valid_cand_outpath = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/read_map_valid_candidates')

#concatenate all the reads from each metagenome for each annotation into one file
for run in runs:
    print(run,'...')
    #get a list of all the files in read_map for this run 
    contents = [x for x in Path(root/run/'read_map').iterdir()]
    #for each of these files, extract the reads, and the labels from the filename
    for fasta in contents:
        records = []
        label = fasta.stem
        for record in SeqIO.parse(fasta,"fasta"): 
            #seq = str(record.seq)
            #record = [run, seq, label]
            records.append(record)
        #write all the records with this annotation from this run to the concatenated file in cdhit_clean_for_training/read_map_all
        with open(outpath/Path(label+'.fasta'),'a') as handle:
            SeqIO.write(records,handle,'fasta')

#for each annotation in read_map_all, select the first 900 reads, or 80% of the number of reads, whichever is bigger, and put in the train folder as train set seqs
#put the rest in the valid_candidate folder
contents = [x for x in outpath.iterdir()]
for annotation in contents:
    print('processing ec number',annotation)
    train_records = []
    valid_records = []
    label = annotation.stem
    for ix,record in enumerate(SeqIO.parse(annotation,"fasta")):
        if ix < 900:
            train_records.append(record)
        else:
            valid_records.append(record)
    
    with open(train_outpath/Path(label+'.fasta'),'a') as handle:
        SeqIO.write(train_records,handle,'fasta')
    with open(valid_cand_outpath/Path(label+'.fasta'),'a') as handle:
        SeqIO.write(valid_records,handle,'fasta')


#with these folders, will compare valid_candidate for a given EC to train set with CD-HIT-EST-2D, 
#keep first 100 seqs (or as many as have if aren't 100) that are nonsimilar to train set as valid seqs
#(see mifasertodatasets_cdhit.sh)




    