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

ecref = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/ecref_valid_genes.csv')
fout = '/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/seqsim_ecs_genes.csv'
ecref_reads = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/ecref_valid.csv')
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
    #get global alignment
    alignment = pairwise2.align.globalms(combo[0],combo[1],1,-1,-10.0,-1.0)
    ix = np.argmax([a[2] for a in alignment]) #use alignment with max score
    #get number of matches
    num_matches = np.sum(np.array(list(alignment[ix][0]))==np.array(list(alignment[ix][1])))
    #divide by length of alignment
    length_alignment = alignment[ix][4]
    percent_identity = num_matches/length_alignment
    max_score = alignment[ix][2]
    return max_score,num_matches,length_alignment,percent_identity

def get_scores(df, ec):
    subset = df[df['EC number']==ec]
    if len(set(subset['Entry']))>1:
        if len(set(subset['Entry']))!=len(subset): #subsample so only one example per uniprot entry
            subset.drop_duplicates(subset='Entry', inplace=True)

        #calculate average within-group accuracy in the validset read-length predictions
        read_subset = ecref_reads[ecref_reads['EC number']==ec]
        acc = np.sum(read_subset['predicted_label']==read_subset['label'])/len(read_subset)

        #calculate within-group identities
        scores = []
        lengths = []
        identities = []
        matches = []
        combos = list(combinations(subset.index,2))
        seq_combos = get_combos(combos, subset, 'seq')
        for combo in seq_combos:
            max_score,num_matches,length_alignment,percent_identity = score_alignment(combo)
            scores.append(max_score)
            lengths.append(length_alignment)
            identities.append(percent_identity)
            matches.append(num_matches)
        mean_score = np.array(scores).mean()
        mean_alignment_length = np.array(lengths).mean()
        mean_percent_identity = np.array(identities).mean()
        mean_num_matches = np.array(matches).mean()
    else:
        read_subset = ecref_reads[ecref_reads['EC number']==ec]
        acc = np.sum(read_subset['predicted_label']==read_subset['label'])/len(read_subset)
        mean_score = np.nan
        mean_alignment_length = np.nan
        mean_percent_identity = np.nan
        mean_num_matches = np.nan

    #return row
    row = [ec,acc,mean_score,mean_num_matches,mean_alignment_length,mean_percent_identity]
    return row


inputs = tqdm(list(set(ecref['EC number'])))
start = datetime.now()
scores = Parallel(n_jobs=num_cpu)(delayed(get_scores)(ecref,ec=i) for i in inputs)
end = datetime.now()
print('took',end-start,'to process',len(set(ecref['EC number'])),'ecs')

print('saving seq similarities to',fout)
with open(fout, 'a') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['EC number','accuracy','bit_score','num_matches','alignment_length','percent_identity'])
    [writer.writerow(x) for x in scores]
