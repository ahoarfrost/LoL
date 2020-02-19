from Bio import SeqIO
import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils import *
from fastai import *
from fastai.text import *

path = Path('./')

train_in = (path / "../data/NearComplete5M_train_seq150.csv")
valid_in = (path / "../data/NearComplete_valid_seq150.csv")
test_in = (path / "../data/NearComplete_test_seq150.csv")
lr_plot_out = 'lr_plot_150bp.png'
batch_size=512
lr = 8e-3
vocab_file = (path / "../vocabs/ngs_vocab_k1.npy").resolve() 
model_name = 'quicklm5M_150bp_k1_morex2'

print('reading data...')
train = pd.read_csv(train_in, names=['location','seq','group'])
valid = pd.read_csv(valid_in, names=['location','seq','group'])

#define our tokenizer
tokk1 = Tokenizer(partial(GenomicTokenizer, ngram=1, stride=1), n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])

voc = np.load(vocab_file)
model_voc = GenomicVocab(voc)

print('creating data bunch...')
start_bunch = datetime.now()
data = GenomicTextLMDataBunch.from_df(path, train, valid, 
                                      bs=batch_size, tokenizer=tokk1, vocab=model_voc,
                                      text_cols='seq')
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to tokenize data')

print('creating LM...')
config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=0, qrnn=False, output_p=0.25, 
                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
drop_mult = 0.3
learn = get_model_LM(data, drop_mult, config)
learn.load('quicklm5M_150bp_k1_more')

print('figuring out LR')
learn.lr_find()
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

print('training 12 cycles...')
start_model = datetime.now()
learn.fit_one_cycle(12, lr, moms=(0.8,0.7))
end_model = datetime.now()
print('...took',str(end_model-start_model),'to train all cycles')

print('saving trained model...')
learn.save(model_name)
learn.save_encoder(model_name+'_encoder')

print('Done!')
