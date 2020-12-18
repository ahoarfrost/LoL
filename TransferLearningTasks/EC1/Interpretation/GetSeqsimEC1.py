#script to get the pairwise sequence similarity and embedding similarity for homologous/nonhomologous pairs 
#(using embedding output of GetReadModelEmbs_singlereads_Homologs_<taxlevel>.py)
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from Bio import pairwise2
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import csv


#example fin: /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_HomEmb_out/OG_SeqEmbs_genus.pkl
#example fout: /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_Similarities_out/Homolog_Emb_Seq_Comp_genus.csv

ecref = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/ecref_valid.csv')
fout = '/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/seqsim_ecs.csv'
num_cpu = 28

def get_combos(combos,df,colname):
    new_combos = []
    for combo in combos:
        a = df.loc[combo[0]][colname]
        b = df.loc[combo[1]][colname]
        new_combos.append((a,b))
    return new_combos

#compute seq similarity (smith waterman bit score) 
def score_alignment(combo):
    alignment = pairwise2.align.localms(combo[0],combo[1],1,-1,-10.0,-1.0)
    ix = np.argmax([a[2] for a in alignment])
    max_score = alignment[ix][2]
    return max_score

def get_scores(df, ec):
    subset = df[df['EC number']==ec]
    if len(subset)>1000:
        subset = subset.sample(1000) #there are 35 EC groups with >1000, subsample to 1000 in that case to keep compute under control
    #calculate average within-group seqsim 
    acc = np.sum(subset['predicted_label']==subset['label'])/len(subset)
    #calculate within-group prediction accuracy    
    scores = []
    combos = list(combinations(subset.index,2))
    seq_combos = get_combos(combos, subset, 'seq')
    for combo in seq_combos:
        seqsim = score_alignment(combo)
        scores.append(seqsim)
    mean_score = np.array(scores).mean()
    #return row
    row = [ec,acc,mean_score]
    return row


inputs = tqdm(list(set(ecref['EC number'])))
start = datetime.now()
scores = Parallel(n_jobs=num_cpu)(delayed(get_scores)(ecref,ec=i) for i in inputs)
end = datetime.now()
print('took',end-start,'to process',len(set(ecref['EC number'])),'ecs')

print('saving seq similarities to',fout)
with open(fout, 'a') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['EC number','accuracy','seqsim'])
    [writer.writerow(x) for x in scores]
