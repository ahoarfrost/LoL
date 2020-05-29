from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import csv
import json

df = pd.read_pickle('/home/ah1114/LanguageOfLife/Interpret/InterpretSmallSubset_Embs.pkl')
aligned = pd.read_csv('/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatchAlignlgth_aligned.csv')
ixes = [json.loads(x) for x in aligned['combo']]
fout = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatchAlignlgthSeqlgth_aligned.csv'

def get_seq_lengths(df,aligned,ixes):
    seqa_lens = []
    seqb_lens = []
    lens = []
    for ix in ixes:
        lena = len(df.iloc[ix[0]]['seq']) 
        lenb = len(df.iloc[ix[1]]['seq']) 
        avg_len = (lena+lenb)/2
        seqa_lens.append(lena)
        seqb_lens.append(lenb)
        lens.append(avg_len)
    return seqa_lens,seqb_lens,lens

def get_ungapped_alignment_length(alignment):
    strand0 = alignment[0][alignment[3]:alignment[4]].split('-')
    strand1 = alignment[1][alignment[3]:alignment[4]].split('-')
    ungapped_length0 = np.sum([len(x) for x in strand0])  
    ungapped_length1 = np.sum([len(x) for x in strand1])  
    ungapped_alignment_length = np.min([ungapped_length0, ungapped_length1])
    return ungapped_alignment_length

def score_alignment(combo):
    alignment = pairwise2.align.localms(combo[0],combo[1],1,-1,-10.0,-1.0)
    ix = np.argmax([a[2] for a in alignment])
    max_score = alignment[ix][2]
    ungapped_alignment_length = get_ungapped_alignment_length(alignment[ix])
    #ungapped_alignment_length = len(alignment[ix][0][alignment[ix][3]:alignment[ix][4]].split('-')[0])  #why would I take just the 0 here?
    return max_score, ungapped_alignment_length

seqa_lens,seqb_lens,lens = get_seq_lengths(df,aligned,ixes)
aligned['seqa_seqlen'] = seqa_lens
aligned['seqb_seqlen'] = seqb_lens
aligned['avg_seqlen'] = lens

alignment_lengths = []
for row,ix in enumerate(ixes):
    if row % 1000 == 0:
        print('processing ix',row,'out of',len(ixes))
    seqa = df.iloc[ix[0]]['seq']
    seqb = df.iloc[ix[1]]['seq']
    score,align_length = score_alignment([seqa,seqb])
    #alignment = pairwise2.align.localms(seqa,seqb,1,-1,-10.0,-1.0)
    alignment_lengths.append(align_length)
    #print('align length',align_length)
    #print(format_alignment(*alignment[0]))

aligned['ungapped_alignment_length_new'] = alignment_lengths
aligned['percent_aligned'] = aligned['ungapped_alignment_length_new']/aligned['avg_seqlen']
aligned.to_csv(fout,index=False)