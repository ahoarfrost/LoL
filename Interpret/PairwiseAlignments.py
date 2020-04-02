from Bio import pairwise2
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

#doing a local alignment 
#with a mismatch penalty (because otherwise weirdo alignments can get kind of high scores), 
#a gap open penalty of -5.0, which is lower than what EMBOSS water uses (-10.0), but seems appropriate for these short sequences where the max alignment length is limited
#a gap extension penalty of -0.5 (1/10 gap open penalty, also same as EMBOSS water)

#e.g. alignment = pairwise2.align.localms(seqa,seqb,1,-1,-5.0,-0.5)
#max_score = np.max([a[2] for a in alignment])

#get all the possible pairs from list of sequences, let's do it for a subset of the mifaser valid set (because there are 16M+ rows and that's too many pairs)
df = pd.read_csv('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/valid/mifaser_valid.csv')
rows = list(range(0,1000)) #let's test the time for subset first
combos = list(itertools.combinations(rows, 2))

def get_seq_combos(combos,df):
    new_combos = []
    for combo in combos:
        seqa = df.iloc[combo[0]]['seq']
        seqb = df.iloc[combo[1]]['seq']
        new_combos.append((seqa,seqb))
    return new_combos

seq_combos = get_seq_combos(combos,df)

#took 39 sec to compute 4950 combinations
#there are 1274 categories in the 'annotation' - if we take 10 from each we have 12740 rows = 81,147,430 combos
#81147430/4950 * 39s/60s/m/60m/h = 177.6h / num_cpus = 6.3h
#100 each cat, 127400 rows = 8,115,316,300 combos => 634h
#there are 95 'runs' from one of 10 env_biomes

#compute alignments in parallel, write to csv
#colnames: (seqid_a, seqid_b), alignment_score
def score_alignment(combo):
    alignment = pairwise2.align.localms(combo[0],combo[1],1,-1,-5.0,-0.5)
    max_score = np.max([a[2] for a in alignment])
    return max_score

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


num_cpu = multiprocessing.cpu_count()
print('num cpu:',num_cpu)
inputs = tqdm(seq_combos)
start = datetime.now()
scores = Parallel(n_jobs=num_cpu)(delayed(score_alignment)(i) for i in inputs)
end = datetime.now()
print('took',end-start,'to process',len(seq_combos),'combinations')