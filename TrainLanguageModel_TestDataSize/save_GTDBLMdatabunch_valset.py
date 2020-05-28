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
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')
data_outfile = 'GTDBdatabunch_validonly.pkl'
#data_outfile = 'GTDBdatabunch_validonly_15kseqs.pkl' #for the 15000 seq limit

#params from parameter search
bs=512 
ksize=1
stride=1
bptt=100 

val_max_seqs=None #this worked on 500GB CPU memory
#there are 41355499 total seqs in 1630 valid files (avg 25371 seqs/file); get Bus error trying to save on 128GB memory; what about smaller amounts?
#val_max_seqs=15000 #this worked on 128GB CPU memory

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)
'''
print('creating validset')
#print('creating validset with 15k max seqs per file')

start_bunch = datetime.now()

processor = [OpenSeqFileProcessor(max_seqs=val_max_seqs, ksize=ksize, skiprows=0)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
data = BioTextList.from_folder(path=valid_path, vocab=model_voc, max_seqs_per_file=val_max_seqs, processor=processor)                                   
trn = data[[]]
trn.ignore_empty = True
data = data._split(data_path,trn,data)
data = data.label_for_lm()
data = data.databunch()

end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('saving databunch')
data.save(data_outfile) #this should save to this file in the data_path
#the full validset saved successfully, and takes up 46G on disk; took about 35min to process

print('Done!')
'''

#note the below doesn't work for 128GB; memory error at point of loading valset. 
#does it work loading valset first then creating train? no.
#(also note only takes 3m 23s to load presaved valset)
#does it work with loading 15k valset? yes.
#does it work with the new 461-genome validset?

#to add train set:
#create databunch with just the training set you want to add
max_seqs=2000 #works with max_seqs=1066; works with 2000?
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
t_processor = [OpenSeqFileProcessor(max_seqs=max_seqs, ksize=ksize, skiprows=0)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
start_bunch = datetime.now()
print('creating training set chunk')
train = BioTextList.from_folder(path=train_path, vocab=model_voc, max_seqs_per_file=max_seqs, processor=t_processor)
val = train[[]]
val.ignore_empty = True
train = train._split(data_path,train,val)
train = train.label_for_lm()
train = train.databunch()
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess training data')
print('there are',len(train.items),'items in itemlist, and',len(train.valid_ds.items),'items in train.valid_ds')
print('loading presaved valid set')                         
#load the saved databunch with the validation set
start_load = datetime.now()
data = load_data(data_path,data_outfile)
end_load = datetime.now()
print('...took',str(end_load-start_load),'to load presaved valid set')
#add the training dataloader to the saved databunch
print('attaching to data')
data.train_dl = train.train_dl
#and now your full databunch is in the variable 'data'
print('Done! Your databunch contains',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
