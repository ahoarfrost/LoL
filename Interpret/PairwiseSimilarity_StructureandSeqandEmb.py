from Bio import pairwise2
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import csv

#doing a local alignment 
#EMBOSS water uses -10.0 gap open penalty and extension penalty -0.5 (and match/mismatch of 1 and 0)
#diamond uses a gap open score of -11 and extension of -1 (and match/mismatch of 1 and 0). only considers an alignment if score >50
#I'm going to use -10, -1, 1, 0

#e.g. alignment = pairwise2.align.localms(seqa,seqb,1,0,-10.0,-1)
#max_score = np.max([a[2] for a in alignment])

#get all the possible pairs from list of sequences, let's do it for a subset of the mifaser valid set (because there are 16M+ rows and that's too many pairs)
#there are 171 rows in  rows in InterpretEColiPDB_Embs.pkl, so can run serially


df = pd.read_pickle('/home/ah1114/LanguageOfLife/Interpret/InterpretEColiPDB_Embs.pkl')
#there are some issues with PDB 6SGG, so remove that
df = df[df['PDBid']!='6SGG']
rows = list(range(0,len(df)))
combos = list(itertools.combinations(rows, 2))
print('processing',len(combos),'combos for',len(rows),'rows in df')

def get_combos(combos,df,colname):
    new_combos = []
    for combo in combos:
        a = df.iloc[combo[0]][colname]
        b = df.iloc[combo[1]][colname]
        new_combos.append((a,b))
    return new_combos

#parse appropriate .tm topmatch outfile for maximum STRSCR
#define max_scores as fixed value; I picked 1000 (in individual topmatch comparison, strscr is length of peptide)
def get_structurescore(combos,df,pdb_combos,same_score=1000):
    scores = []
    scoredict = {}
    for ix,combo in enumerate(combos):
        if ix % 10000 == 0:
            print('processing combo',ix,'out of',len(combos))
        pdba = df.iloc[combo[0]]['PDBid']
        pdbb = df.iloc[combo[1]]['PDBid']
        if pdba==pdbb:
            #perfect match for self comparison - pick max as 1000 because maximum non-same score is 423
            score = int(same_score)
        else:
            #get filename where topmatch score is
            comboname = pdba+str('_')+pdbb
            filename = pdb_combos[comboname]
            if filename:
                try:
                    #check if already got score, grab from dict if so
                    if comboname in scoredict.keys():
                        score = scoredict[comboname]
                    else: #if not parse it and add to scoredict
                        record = parse_topmatch(filename)
                        score = np.max([int(x.STRSCR) for x in record.rank.values()])
                        scoredict[comboname] = score
                except:
                    print('parse error for',filename)
                    score = int(0)
                    scoredict
                    continue
            else:
                print('no file exists for',comboname)
        scores.append(score)
    return scores

files = [p for p in pathlib.Path('/home/ah1114/LanguageOfLife/Interpret/short_outs').iterdir() if p.is_file()]
pdb_combos = {}
for x in files:
    pdb_combos['_'.join([x.name.split('_')[0],x.name.split('_')[2]])] = str(x)

time_start = datetime.now()
structure_scores = get_structurescore(combos,df,pdb_combos)
time_end = datetime.now()
print('took',(time_end-time_start),'to get structure scores for',len(combos),'combos')

seq_combos = get_combos(combos,df,'seq')
emb_combos = get_combos(combos,df,'emb')
all_combos = list(zip(combos,seq_combos,emb_combos,structure_scores))
print('number of combos to compute:',len(all_combos))


#compute alignments in parallel, write to csv
#colnames: (seqid_a, seqid_b), alignment_score
def score_alignment(combo):
    alignment = pairwise2.align.localms(combo[0],combo[1],1,0,-10.0,-1.0)
    max_score = np.max([a[2] for a in alignment])
    return max_score

#compute cosine similarity manually
def cosine(combo):
    a=combo[0]
    b=combo[1]
    dot = np.dot(a,b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma*normb)
    return cos

def get_scores(combos):
    combo = list(combos[0])
    alignment_score = score_alignment(combos[1])
    cosine_similarity = cosine(combos[2])
    structure_score = combos[3]
    return [combo,alignment_score,cosine_similarity,structure_score]

fout = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretEColiPDB.csv'
num_cpu = multiprocessing.cpu_count()
print('computing on num cpu:',num_cpu)
inputs = tqdm(all_combos)
start = datetime.now()
scores = Parallel(n_jobs=num_cpu)(delayed(get_scores)(i) for i in inputs)
end = datetime.now()
print('took',end-start,'to process',len(seq_combos),'combinations')
print('saving')
with open(fout, 'a') as outfile:
    writer = csv.writer(outfile)
    #writer.writerow(['combo','seq_similarity','emb_similarity'])
    writer.writerows(scores)

