#script to get the pairwise sequence similarity and embedding similarity for homologous/nonhomologous pairs 
#(using embedding output of GetReadModelEmbs_singlereads_Homologs_<taxlevel>.py)
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations, cycle, islice
from Bio import pairwise2
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fin",type=str)
parser.add_argument("--fout",type=str)
args = parser.parse_args()

#example fin: /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_HomEmb_out/OG_SeqEmbs_genus.pkl
#example fout: /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_Similarities_out/Homolog_Emb_Seq_Comp_genus.csv

print('calculating seq and emb similarities for', args.fin)
emb = pd.read_pickle(args.fin)
fout = args.fout
num_cpu = multiprocessing.cpu_count()

def get_combos(combos,df,colname):
    new_combos = []
    for combo in combos:
        a = df.iloc[combo[0]][colname]
        b = df.iloc[combo[1]][colname]
        new_combos.append((a,b))
    return new_combos

#compute cosine similarity manually
def cosine(combo):
    a=combo[0]
    b=combo[1]
    dot = np.dot(a,b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma*normb)
    return cos

def get_ungapped_alignment_length(alignment):
    strand0 = alignment[0][alignment[3]:alignment[4]].split('-')
    strand1 = alignment[1][alignment[3]:alignment[4]].split('-')
    ungapped_length0 = np.sum([len(x) for x in strand0])  
    ungapped_length1 = np.sum([len(x) for x in strand1])  
    ungapped_alignment_length = np.min([ungapped_length0, ungapped_length1])
    return ungapped_alignment_length

#compute seq similarity (smith waterman bit score) 
def score_alignment(combo):
    alignment = pairwise2.align.localms(combo[0],combo[1],1,-1,-10.0,-1.0)
    ix = np.argmax([a[2] for a in alignment])
    max_score = alignment[ix][2]
    #ungapped_alignment_length = get_ungapped_alignment_length(alignment[ix])
    return max_score

def get_cat_combos(df,og):
    subset = df[df['og']==og]
    ix = list(subset.index)
    #get in-group combos
    hom_combos = list(combinations(ix,2))
    #choose random indeces not in ix to compare to, same number as hom_combos
    outgroup = list(df.index)
    del outgroup[ix[0]:ix[-1]+1]
    inog = list(islice(cycle(ix), len(hom_combos)))
    outog = list(np.random.choice(outgroup,len(hom_combos),replace=False))
    out_combos = list(zip(inog,outog))

    return hom_combos,out_combos

def get_scores(emb,og):
    scores = []
    in_combos,out_combos = get_cat_combos(emb,og=og)
    
    new_in_combos = get_combos(in_combos,emb,'emb')
    seq_in_combos = get_combos(in_combos,emb,'seq')
    for ix,combo in enumerate(new_in_combos):
        cos = cosine(combo)
        seq_combo = seq_in_combos[ix]
        seqsim = score_alignment(seq_combo)
        row = [og,in_combos[ix][0],in_combos[ix][1],'Homologous',cos,seqsim]
        scores.append(row)

    new_out_combos = get_combos(out_combos,emb,'emb')
    seq_out_combos = get_combos(out_combos,emb,'seq')
    for ix,combo in enumerate(new_out_combos):
        cos = cosine(combo)
        seq_combo = seq_out_combos[ix]
        seqsim = score_alignment(seq_combo)
        row = [og,out_combos[ix][0],out_combos[ix][1],'Nonhomologous',cos,seqsim]
        scores.append(row)
    return scores


inputs = tqdm(list(set(emb['og'])))
all_scores = []
start = datetime.now()
scores = Parallel(n_jobs=num_cpu)(delayed(get_scores)(emb,og=i) for i in inputs)
end = datetime.now()
print('took',end-start,'to process',len(set(emb['og'])),'ogs')

print('saving seq and embedding similarities to',fout)
with open(fout, 'a') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['og','a_ix','b_ix','homolog','cos','seqsim'])
    [writer.writerows(x) for x in scores]
