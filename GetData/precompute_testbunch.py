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
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')
outpath = Path('/scratch/ah1114/LoL/data/')

#params from parameter search
bs=512 
ksize=1
stride=1
bptt=100 

max_seqs=10 #this will total about 20-25M seqs in databunch
val_max_seqs=10 #this is about 11M seqs

skiprows=0

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

data_outfile = 'GTDBdatabunch_fortesting.pkl'

start_bunch = datetime.now()
data = BioLMDataBunch.from_folder(path=outpath, 
                                        train=train_path, valid=valid_path, ksize=ksize,
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, val_maxseqs=val_max_seqs,
                                        skiprows=skiprows, val_skiprows=0, bs=bs, bptt=bptt
                                            )
end_bunch = datetime.now()
print('took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'training items, and',len(data.valid_ds.items),'validation items')

data.device = torch.device('cuda')
print('device is',data.device)
print('saving databunch')
start_save = datetime.now()
data.save(data_outfile)
end_save = datetime.now()
print('took', str(end_save-start_save),'to save data')