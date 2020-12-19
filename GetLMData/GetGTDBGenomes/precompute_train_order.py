#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime
import gc 

path = Path('./') 
data_path = Path('/scratch/ah1114/LoL/data/')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train_parsedorder/')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid_parsedorder/')
data_outfile = 'GTDB_trainorder_databunch.pkl'
device = torch.device('cuda')
#params
bs=512 
ksize=1
stride=1
bptt=100

max_seqs=None
val_max_seqs=None 
skiprows = 0
val_skiprows = 0

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

#create new training chunk 
print('creating databunch')
start_bunch = datetime.now()
data = BioLMDataBunch.from_folder(path=data_path, 
                                        train=train_path, valid=valid_path, ksize=ksize,
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, val_maxseqs=val_max_seqs,
                                        skiprows=skiprows, val_skiprows=val_skiprows,
                                        bs=bs, bptt=bptt
                                            )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
#make sure device is where it should be
data.device = device
print('youre on device:',data.device)
print('databunch preview:')
print(data)
data.save(data_outfile)