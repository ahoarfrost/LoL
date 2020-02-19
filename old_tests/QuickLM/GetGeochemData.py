import csv
from Bio import SeqIO
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys

from utils import *
from fastai import *
from fastai.text import *

path = Path('./')

#load our LM
print('loading LM...')
def split_data(df, pct, random_state):
    import pandas as pd
    
    train = df.sample(frac=pct, random_state=random_state)
    valid = df.drop(train.index)
    
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    
    return train, valid

mini_file = (path / "../../test/MiniData_Genomic_complements.csv").resolve() 
mini = pd.read_csv(mini_file, names=["location","frame","sequence"])
train, valid = split_data(mini, pct=0.9, random_state=0)

#define our tokenizer and load our saved vocab from Data_Processing notebook
vocab_file = (path / "../vocabs/ngs_vocab_k1.npy").resolve() 
voc = np.load(vocab_file)
model_voc = GenomicVocab(voc)
tok = Tokenizer(partial(GenomicTokenizer, ngram=1, stride=1), n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])

data = GenomicTextLMDataBunch.from_df(path, train, valid, 
                                      bs=512, tokenizer=tok, vocab=model_voc,
                                      text_cols='sequence')

config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
drop_mult = 0.3
learn = get_model_LM(data, drop_mult, config)
learn.load('quicklm5M_150bp_k1_morex2')

def get_embed(trn_example, vocab, model): #e.g. learn.data.train_ds[0][0], learn.data.vocab, learn.model
    tokens = list(trn_example) #this only works for k1
    nums = vocab.numericalize(tokens)
    embed = model[0].encoder(torch.from_numpy(np.array(nums)).to('cuda'))
    mean_embed = embed.mean(0).data.cpu().numpy()
    return mean_embed

small_ERRs = [
'/scratch/ah1114/LoL/data/ERR315856_1.fastq',
'/scratch/ah1114/LoL/data/ERR315857_1.fastq',
'/scratch/ah1114/LoL/data/ERR315862_1.fastq',
'/scratch/ah1114/LoL/data/ERR318618_1.fastq',
'/scratch/ah1114/LoL/data/ERR594288_1.fastq',
'/scratch/ah1114/LoL/data/ERR594294_1.fastq',
'/scratch/ah1114/LoL/data/ERR594299_1.fastq',
'/scratch/ah1114/LoL/data/ERR594310_1.fastq',
'/scratch/ah1114/LoL/data/ERR594313_1.fastq',
'/scratch/ah1114/LoL/data/ERR594315_1.fastq',
'/scratch/ah1114/LoL/data/ERR594318_1.fastq',
'/scratch/ah1114/LoL/data/ERR594321_1.fastq',
'/scratch/ah1114/LoL/data/ERR594336_1.fastq',
'/scratch/ah1114/LoL/data/ERR594340_1.fastq',
'/scratch/ah1114/LoL/data/ERR594349_1.fastq',
'/scratch/ah1114/LoL/data/ERR598945_1.fastq',
'/scratch/ah1114/LoL/data/ERR598950_1.fastq',
'/scratch/ah1114/LoL/data/ERR598951_1.fastq',
'/scratch/ah1114/LoL/data/ERR598955_1.fastq',
'/scratch/ah1114/LoL/data/ERR598993_1.fastq',
'/scratch/ah1114/LoL/data/ERR599073_1.fastq',
'/scratch/ah1114/LoL/data/ERR599094_1.fastq',
]

ERRs = [
    '/scratch/ah1114/LoL/data/ERR315856_1.fastq',
    '/scratch/ah1114/LoL/data/ERR315857_1.fastq',
    '/scratch/ah1114/LoL/data/ERR315862_1.fastq',
    '/scratch/ah1114/LoL/data/ERR318618_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594288_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594294_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594299_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594310_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594313_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594315_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594315_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594318_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594318_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594321_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594336_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594340_1.fastq',
    '/scratch/ah1114/LoL/data/ERR594349_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598945_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598947_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598949_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598950_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598951_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598955_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598959_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598960_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598965_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598966_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598967_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598970_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598972_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598973_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598974_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598975_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598977_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598979_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598982_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598984_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598990_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598993_1.fastq',
    '/scratch/ah1114/LoL/data/ERR598994_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599000_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599002_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599005_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599006_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599009_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599010_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599011_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599013_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599017_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599019_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599021_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599024_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599026_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599027_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599031_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599037_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599040_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599041_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599042_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599044_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599045_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599046_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599049_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599050_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599057_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599058_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599061_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599064_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599071_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599073_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599075_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599081_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599090_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599093_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599094_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599098_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599102_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599104_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599109_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599112_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599124_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599129_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599133_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599135_1.fastq',
    '/scratch/ah1114/LoL/data/ERR599145_1.fastq'
]

file_in = sys.argv[1]
out = 'TARAavgembed_med2M.csv'

with open(out, 'a') as csvfile:
    fin = file_in
    prefix = fin.split('_1.fastq')[0][-9:]
    print('processing',prefix)
    num_incomplete = 0
    num_extracted = 0
    embeddings = np.zeros(400)
    ix = 0
    for fq in SeqIO.parse(fin,'fastq'):
        if ix==2500000:
            break
        if set(fq.seq)!=set('ATGC'): #ignore any sequences with unknown base pairs
            num_incomplete += 1
            continue
        else:
            num_extracted += 1
            embed = get_embed(fq.seq, learn.data.vocab, learn.model)
            embeddings = np.array(((embeddings*ix)+embed))/(ix+1)
            #embeddings.append(embed)
            ix += 1

    doc_embedding = embeddings #np.array(embeddings).mean(0)
    data = [prefix]+list(doc_embedding)
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(data)
