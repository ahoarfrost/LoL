#count the majority class accuracy for k1 and k3
import sys
sys.path.insert(0, '/home/ah1114/LanguageOfLife/BioDL')
#and import all the stuff
from data import *
from Bio import SeqIO

train = '/scratch/ah1114/LoL/data/ModelSelectionMiniset/train/GTDBrepGenomes10_train.fna'
k1voc = '/home/ah1114/LanguageOfLife/vocabs/ngs_vocab_k1_withspecial.npy'
k3voc = '/home/ah1114/LanguageOfLife/vocabs/ngs_vocab_k3_withspecial.npy'
itos1 = list(np.load(k1voc)[4:])
itos3 = list(np.load(k3voc)[4:])
counts1 = dict.fromkeys(itos1,0) 
counts3 = dict.fromkeys(itos3,0) 

tok1 = BioTokenizer(ksize=1,stride=1)
tok3 = BioTokenizer(ksize=3,stride=1)

for record in SeqIO.parse(train,"fasta"): 
    print(record)
    print(record.seq)
    tokens1 = tok1.tokenizer(str(record.seq), include_bos=False)
    try:
        tokens3 = tok3.tokenizer(str(record.seq), include_bos=False)
    except IndexError:
        continue
    for token1 in tokens1:
        try:
            counts1[token1] += 1
        except KeyError:
            continue
    for token3 in tokens3:
        try:
            counts3[token3] += 1
        except KeyError:
            continue

df1 = pd.DataFrame.from_dict(counts1, orient="index",columns=['count']) 
df3 = pd.DataFrame.from_dict(counts3, orient="index",columns=['count']) 
df1.to_csv('k1count.csv')
df3.to_csv('k3count.csv')