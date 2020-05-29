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
classes = list(np.load('/home/ah1114/LanguageOfLife/MifaserClassification/mifaser_classes.npy'))
ec1 = [x for x in classes if x.startswith('1.')]
ec1_3 = ['.'.join(x.split('.')[0:3]) for x in ec1] 
ec1_3 = list(set(ec1_3))
print('number of ec1 classes at level 3:',len(ec1_3))

valid = pd.read_csv('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/valid/mifaser_valid.csv')
valid['annotation_level3'] = ['.'.join(x.split('.')[0:3]) for x in valid['annotation']]
ec1df = valid[valid['annotation_level3'].str.startswith('1.')]
print('number of ec1 classes at level 3 in ec1df',len(set(ec1df['annotation_level3'])))
annotations = list(set(ec1df['annotation_level3']))

env = pd.read_csv('TrainSmall95_EvenEnv.csv') 
env.index = env['run_ids_maxrun']  
ec1df = ec1df.join(env[['metaseek_env_package']],on='run')  

#go through each ERR accession; for each mifaser pull out 10 seqs for each annotation
#this is the 94 runs identified in GetWGSmetagenomes as (not duplicates) found in validset
matchruns = ['ERR1855552', 'ERR2017141', 'SRR925829', 'ERR2239841', 'ERR2145413', 'SRR1298754', 'ERR1190830', 'ERR970605', 'ERR2241851', 
'SRR4069407', 'ERR1877921', 'ERR2144884', 'ERR2020015', 'ERR1739691', 'SRR1238204', 'ERR1743388', 'ERR3521978', 'ERR1726943', 'ERR977414', 
'ERR598955', 'SRR5164026', 'ERR1743304', 'ERR2699809', 'ERR1135232', 'ERR2239851', 'ERR1960627', 'ERR1855546', 'ERR1358726', 'ERR209703', 
'SRR1185414', 'ERR1726572', 'ERR2709750', 'ERR3053395', 'ERR2200674', 'ERR2603191', 'ERR1743283', 'SRR1171647', 'ERR1746303', 'SRR830624', 
'ERR712383', 'ERR2239840', 'ERR2298558', 'ERR209626', 'ERR2022395', 'ERR1474613', 'SRR1577782', 'ERR694158', 'ERR1332606', 'ERR2699805', 
'ERR2765138', 'ERR2969987', 'ERR977422', 'SRR1574704', 'ERR1855538', 'SRR5091465', 'ERR1698988', 'ERR1366724', 'ERR1076075', 
'ERR1201181', 'SRR2556884', 'ERR599252', 'ERR2215874', 'DRR027592', 'ERR2709726', 'ERR1700691', 'SRR1511001', 'ERR2819892', 'ERR1332600', 
'ERR2200504', 'SRR4343439', 'ERR1017187', 'ERR1960504', 'ERR1332623', 'SRR2938315', 'ERR1353140', 'SRR1707409', 'ERR1076080', 'ERR1726775', 
'ERR2145397', 'ERR3173383', 'ERR1743339', 'ERR1332586', 'SRR1616983', 'SRR1577774', 'ERR1332614', 'ERR2816294', 'ERR2767288', 
'ERR2020026', 'ERR1560098', 'SRR1754159', 'ERR1600429', 'SRR5234512']       
print('number of runs to match:',len(matchruns))

num_to_subset = 10
notenough=0
equal_subset = pd.DataFrame()
for run in matchruns: #94 runs
    #get the subset of just this run
    run_subset = ec1df[ec1df['run']==run]
    print('num ECs in run:',run,':',len(set(run_subset['annotation_level3'])))
    #get the subset of annotation
    for anno in annotations: #74 annotations
        anno_subset = run_subset[run_subset['annotation_level3']==anno]
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

equal_subset.to_csv('InterpretSmallSubsetSeqs.csv',index=False)

#try getting an even annotation distribution; see how off this is with env_package
equal_anno_subset = pd.DataFrame()
num_to_subset=1000
notenougha=0
for anno in annotations:
    subset = ec1df[ec1df['annotation_level3']==anno]
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
print(Counter(equal_anno_subset['annotation_level3']).most_common())
#pretty uneven for env, annotation looks good...
#only 8 annotations less than 900...
equal_anno_subset.to_csv('InterpretSubsetEvenAnno.csv',index=False)

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
    