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
#EMBOSS water uses -10.0 gap open penalty and extension penalty -0.5 (and match/mismatch of 1 and -1)
#diamond uses a gap open score of -11 and extension of -1 (and match/mismatch of 1 and -1). only considers an alignment if score >50
#I'm going to use -10, -1, 1, -1

#e.g. alignment = pairwise2.align.localms(seqa,seqb,1,0,-10.0,-1)
#max_score = np.max([a[2] for a in alignment])

#get all the possible pairs from list of sequences, let's do it for a subset of the mifaser valid set (because there are 16M+ rows and that's too many pairs)
#there are 171 rows in  rows in InterpretEColiPDB_Embs.pkl, so can run serially



df = pd.read_pickle('/home/ah1114/LanguageOfLife/Interpret/InterpretSmallSubset_Embs.pkl')
'''
#split rows of df into chunks to process in separate slurm jobs
rows = list(range(0,len(df))) #let's test the time for subset first
combos = list(itertools.combinations(rows, 2))
#split in 8, write to file
n_chunks = 8
chunk_size=int(len(combos)/n_chunks)+1
chunked = [combos[i:i + chunk_size] for i in range(0, len(combos), chunk_size)]
for ix,chunk in enumerate(chunked):
    filename='combo_chunk'+str(ix)+'.npy'
    print('saving', filename)
    np.save(filename,np.array(chunk))

#and get the output csv file ready with header
fout = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset.csv'
with open(fout, 'a') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['combo','seq_similarity','emb_similarity'])
'''
chunk = int(sys.argv[1])
filename='combo_chunk'+str(chunk)+'.npy'
print('getting combos from',filename)
combos = np.load(filename)

def get_combos(combos,df,colname):
    new_combos = []
    for combo in combos:
        a = df.iloc[combo[0]][colname]
        b = df.iloc[combo[1]][colname]
        new_combos.append((a,b))
    return new_combos

seq_combos = get_combos(combos,df,'seq')
emb_combos = get_combos(combos,df,'emb')
all_combos = list(zip(combos,seq_combos,emb_combos))
print('number of combos to compute:',len(all_combos))
 
'''
#took 39 sec to compute seq similarity on 4950 combinations
#took 2:20 m:s to process 10000 combinations
#there are 1274 categories in the 'annotation' - if we take 10 from each we have 12740 rows = 81,147,430 combos
#81147430/4950 * 39s/60s/m/60m/h = 177.6h / num_cpus = 6.3h
#100 each cat, 127400 rows = 8,115,316,300 combos => 634h
#there are 74 categories in the 'annotation_level3'
#there are 95 'runs' from one of 10 env_biomes
#48000 rows = 1151976000 combos => 160h on 28 cpus, 20h each in 8 jobs

how about doing pairwise cosine similarity of embeddings?
for 48000 rows, 1,151,976,000 combos
took 0.085sec to compute 4950 combos - took 8s to compute 499500 combinations
so on 28 cpus will take ~11min - easy
'''

#compute alignments in parallel, write to csv
#colnames: (seqid_a, seqid_b), alignment_score
def score_alignment(combo):
    alignment = pairwise2.align.localms(combo[0],combo[1],1,-1,-10.0,-1.0)
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
    return [combo,alignment_score,cosine_similarity]

'''
#single cpu processing
print('computing pairwise alignments for',len(combos),'combinations')
start = datetime.now()
comb = []
scores = []
for combo in seq_combos:
    max_score = score_alignment(combo)
    comb.append(combo)
    scores.append(max_score)
end = datetime.now()
print('took',end-start,'to process',len(seq_combos),'combinations')
'''

fout = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatch.csv'
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

