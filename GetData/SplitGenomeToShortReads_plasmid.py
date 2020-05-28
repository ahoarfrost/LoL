from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from pathlib import Path
import random
import csv 

train_files = list(set(pd.read_csv('/home/ah1114/LanguageOfLife/GetData/genomes_with_plasmids_genusparsed.txt',header=None)[0]))
valid_files = list(set(pd.read_csv('/home/ah1114/LanguageOfLife/GetData/genomes_with_plasmids_genusparsed_valid.txt',header=None)[0]))
print('number of train genomes:',len(train_files),'; number of valid genomes:',len(valid_files))
distrib = np.load('/scratch/ah1114/LoL/data/GTDBGenomeSequencingDistrib.npy')

def separate_plasmid(fname):
    plasmids = []
    chromosomes = []
    for fna in SeqIO.parse(fname,"fasta"):
        if 'plasmid' in fna.description:
            plasmids.append(fna)
        else:
            chromosomes.append(fna)   
        if len(chromosomes)>1:
            print('unsure for fname',fname,', skipping') #37 of these ot of 378, just skip em
            return [],[]
    return plasmids,chromosomes

def check_plasmids(files):
    count = 0
    for fname in files:
        plasmids,chromosomes = separate_plasmid(fname)
        if len(chromosomes)>1:
            count += 1
            print('unsure for fname',fname)
            print(len(plasmids),len(chromosomes))
            print('plasmid sizes:',[len(p.seq) for p in plasmids])
            print('chromosome sizes:',[len(c.seq) for c in chromosomes])  
    print(count,'genomes with multiple chromosome scaffolds (unknown plasmid or not?)')

#chunks sequence into read-length chunks, writes to csv file
def chunk_sequence(fna, label, genome_name, distrib, shuffle=True):
    #fin should be a fasta file you want to process
    #fout is the name of the clean csv file you want to put into your model
    #distrib is an array of sequence lengths that reflect the distribution of seqlens you want in your output
    start = 0
    records = []
    while start < len(fna.seq):
        seqlen = int(np.random.choice(distrib))
        #IRL, gene is equally likely to be sequenced in either forward or reverse direction. Toss a coin which to use
        use_complement = np.random.choice([0,1])
        if use_complement:
            sequence = fna.seq[start:start+seqlen]
            subseq = sequence.reverse_complement()
            #create name for csv
            subdescription = "[subset=reverse"+str(start)+".."+str(start+seqlen)+"]"
        else:
            subseq = fna.seq[start:start+seqlen]
            subdescription = "[subset="+str(start)+".."+str(start+seqlen)+"]"
        record = [genome_name, fna.description, str(subseq), subdescription, label]
        records.append(record)  
        start = start + seqlen
    #write line to fasta fout
    if shuffle:
        #shuffle the records first
        np.random.shuffle(records)
    return records
    

vout = '/scratch/ah1114/LoL/data/PlasmidChromosomal_genusparsed_valid.csv'
with open(vout, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['genome','description','seq','position','plasmid_chromosome'])
    for fname in valid_files:
        fname = Path(fname)
        genome_name = fname.name.split('_genomic.fna')[0] 
        plasmids,chromosomes = separate_plasmid(fname)
        num_plasmid_inputs = 0
        for p in plasmids:
            p_records = chunk_sequence(p, label='plasmid',genome_name=genome_name, distrib=distrib, shuffle=True)
            num_plasmid_inputs += len(p_records)
            writer.writerows(p_records)
        for c in chromosomes:
            c_records = chunk_sequence(c, label='chromosome',genome_name=genome_name, distrib=distrib, shuffle=True)
            subset_c_records = random.sample(c_records, num_plasmid_inputs) 
            writer.writerows(subset_c_records)

#same for train
tout = '/scratch/ah1114/LoL/data/PlasmidChromosomal_genusparsed_train.csv'
with open(tout, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['genome','description','seq','position','plasmid_chromosome'])
    for fname in train_files:
        fname = Path(fname)
        genome_name = fname.name.split('_genomic.fna')[0] 
        plasmids,chromosomes = separate_plasmid(fname)
        num_plasmid_inputs = 0
        for p in plasmids:
            p_records = chunk_sequence(p, label='plasmid',genome_name=genome_name, distrib=distrib, shuffle=True)
            num_plasmid_inputs += len(p_records)
            writer.writerows(p_records)
        for c in chromosomes:
            c_records = chunk_sequence(c, label='chromosome',genome_name=genome_name, distrib=distrib, shuffle=True)
            #balance classes
            subset_c_records = random.sample(c_records, num_plasmid_inputs) 
            writer.writerows(subset_c_records)