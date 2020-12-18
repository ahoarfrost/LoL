'''
selecting subset of sequences from 1000 metagenomes with even distributions of function and biome

how long does it take to get embeddings for, say, 1000 sequences (single read at a time)? took 19min in my test

I ran this in an interactive session, so output csv should exist in Interpret/ folder

'''

import pandas as pd 
import numpy as np
from collections import Counter 
import csv

#define mifaser classes
classes = list(np.load('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/mifaser_classes.npy'))
ec1 = [x for x in classes if x.startswith('1.')]
ec1_3 = ['.'.join(x.split('.')[0:3]) for x in ec1] 
ec1_3 = list(set(ec1_3))
print('number of ec1 classes at level 3:',len(ec1_3))

valid = pd.read_csv('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/valid/mifaser_valid.csv')
#valid['annotation_level3'] = ['.'.join(x.split('.')[0:3]) for x in valid['annotation']]
#ec1df = valid[valid['annotation_level3'].str.startswith('1.')]
print('number of ec1 classes at level 4',len(set(valid['annotation'])))
annotations = list(set(valid['annotation']))

env = pd.read_csv('TrainSmall95_EvenEnv.csv') 
env.index = env['run_ids_maxrun']  
anndf = valid.join(env[['metaseek_env_package']],on='run')  
anndf = anndf.dropna()

#go through each ERR accession; for each mifaser pull out 10 seqs for each annotation
#this is the 94 runs identified in GetWGSmetagenomes as (not duplicates) found in validset
matchruns = list(set(anndf['run']))      
print('number of runs to match:',len(matchruns))

num_to_subset = 10
notenough=0
equal_subset = pd.DataFrame()
for run in matchruns: #94 runs
    #get the subset of just this run
    run_subset = anndf[anndf['run']==run]
    print('num ECs in run:',run,':',len(set(run_subset['annotation'])))
    #in this run get num_to_subset reads of each annotation 
    for anno in annotations: #1274 annotations
        anno_subset = run_subset[run_subset['annotation']==anno]
        if len(anno_subset)<num_to_subset:
            notenough +=1
            equal_subset = equal_subset.append(anno_subset)
        else:
            #take random subset rows
            sample = anno_subset.sample(n=num_to_subset)
            equal_subset = equal_subset.append(sample)

print('number of rows in equal_subset:',len(equal_subset)) #43794 rows
print('number of run/anno combos with fewer than num_to_subset to add:',notenough)
print('number of total possible run/anno combos:',len(annotations)*len(matchruns))
print(equal_subset.head())
print('count of env packages:')
print(Counter(equal_subset['metaseek_env_package']).most_common())

equal_subset.to_csv('/scratch/ah1114/LoL/InterpretLG/InterpretSubsetEvenEnv.csv',index=False)

#try getting an even annotation distribution; see how off this is with env_package
equal_anno_subset = pd.DataFrame()
num_to_subset=1000
notenougha=0
for anno in annotations:
    subset = anndf[anndf['annotation']==anno]
    if len(subset)<num_to_subset:
        notenougha+=1
        equal_anno_subset = equal_anno_subset.append(subset)
    else:
        sample = subset.sample(n=num_to_subset)
        equal_anno_subset = equal_anno_subset.append(sample)

equal_anno_subset = equal_anno_subset.dropna()
print('length equal_anno_subset:',len(equal_anno_subset))
print('max possible length:',num_to_subset*len(annotations))
print('number annotations with fewer than num_to_subset examples:',notenougha)
print(equal_anno_subset.head())
print('count of env packages:')
print(Counter(equal_anno_subset['metaseek_env_package']).most_common())
print('count of annotations:')
print(Counter(equal_anno_subset['annotation']).most_common())
#pretty uneven for env, annotation looks good...
#only 8 annotations less than 900...
equal_anno_subset.to_csv('/scratch/ah1114/LoL/InterpretLG/InterpretSubsetEvenAnno.csv',index=False)

'''
#what if I just do a random subset?
random_subset = ec1df.sample(n=100000).dropna()
#even more uneven, for both env and anno; not going to save this

dropped = equal_anno_subset.copy()
[dropped.drop(ix,inplace=True) for ix in equal_subset.index if ix in dropped.index]
#how uneven is it if you append the dropped rows to add to equal_subset?
added = equal_subset.append(dropped)
#env package get quite uneven (counter range 5497-17131); annotation is more uneven but not terrible (counter range 89-1916 with all but ten annotations >1000)
#eh not going to save
'''

'''
#select aligned sequences from /scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatchAlignlgth.csv
fout = "/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatchAlignlgth_aligned.csv"
fin = "/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatchAlignlgth.csv"
with open(fout,'w') as outfile:
    fieldnames = ['combo','seq_similarity','ungapped_alignment_length','emb_similarity']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    with open(fin,'r') as infile:
        reader = csv.DictReader(infile)
        for ix,row in enumerate(reader):
            if ix % 10000000 == 0:
                print('processing row #',ix)
            if row['seq_similarity'] >= 50:
                writer.writerow(row)
'''