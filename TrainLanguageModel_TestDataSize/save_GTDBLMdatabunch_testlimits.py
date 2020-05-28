#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime
import gc

path = Path('./') 
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/')
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')
data_outfile = 'testbunchout_onechunk.pkl'

#params from parameter search
bs=512 
ksize=1
stride=1
bptt=100 

max_seqs=250 #this will total about 25M seqs in databunch, 2.5M in valid rest in train; going to need to do the shuffled batches of files or something to do more than this!
val_max_seqs=None #10000 works, 15000 works, 18000 works, 20000 doesnt, None doesn't (average 22333 seqs/file)
#62705486 total seqs works (1000t18000v)
#with the 40M validation set, 1066 in training doesn't work; 500 doesn't work;
print('test max_seqs',max_seqs)

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

print('creating databunch')

start_bunch = datetime.now()
data = BioLMDataBunch.from_folder(path=data_path, 
                                        train=train_path, valid=valid_path, ksize=ksize,
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, val_maxseqs=val_max_seqs,
                                        skiprows=0, val_skiprows=0, bs=bs, bptt=bptt
                                            )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('saving databunch')
data.save(data_outfile) #this should save to this file in the data_path

print('Done!')