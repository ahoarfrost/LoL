import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
import gc
import pickle

path = Path('./') 
pretrained_start = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_loadsingle')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/')
presaved_validset = 'GTDBdatabunch_validonly.pkl'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/lrplot_GTDB_read_LM.png'
test_path = '/scratch/ah1114/LoL/data/testGenomes/'
#train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
#valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')

device = torch.device('cuda')

#params from parameter search
bs=512 
ksize=1
stride=1
n_layers=3
n_hid=1152
drop_mult=0.2
wd=1e-3
moms=(0.95,0.85)
emb_sz=100
bptt=100
#lrate=8e-3 
lrate=2e-3

max_seqs=5 #this will total about 20M seqs in databunch
val_max_seqs=None #this will be 11M seqs
num_cycles=1

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

skiprows = 0
val_skiprows = 0

#create new training chunk
t_processor = [OpenSeqFileProcessor(max_seqs=max_seqs, ksize=ksize, skiprows=skiprows)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
start_bunch = datetime.now()
print('creating training set chunk')
data = BioTextList.from_folder(path=test_path, vocab=model_voc, max_seqs_per_file=max_seqs, processor=t_processor)
data = data.process()
print(len(data.items))
#save
pickle.dump(data, open( "/scratch/ah1114/LoL/data/test_itemlist1.pkl", "wb" ) )
#test load
load = pickle.load( open( "/scratch/ah1114/LoL/data/test_itemlist1.pkl", "rb" ) )
#create a second training chunk
skiprows=5
t_processor = [OpenSeqFileProcessor(max_seqs=max_seqs, ksize=ksize, skiprows=skiprows)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
train2 = BioTextList.from_folder(path=test_path, vocab=model_voc, max_seqs_per_file=max_seqs, processor=t_processor)
train2 = train2.process()
print(len(train2.items))
#save
pickle.dump(train2, open( "/scratch/ah1114/LoL/data/test_itemlist2.pkl", "wb" ) )
load2 = pickle.load( open( "/scratch/ah1114/LoL/data/test_itemlist2.pkl", "rb" ) )

#join them
data = data.add(train2)
print(len(data.items))

# finish creating databunch
val = data[[]]
val.ignore_empty = True
data = data._split(data_path,data,val)
#train = train.label_for_lm() #this fails b/c tries to process again
data.train = data.train.label_for_lm(from_item_lists=True)
data.valid = data.valid.label_for_lm(from_item_lists=True)
data.__class__ = LabelLists
data = data.databunch()
end_bunch = datetime.now() 
print('...took',str(end_bunch-start_bunch),'to preprocess training data')
print('there are',len(data.items),'items in training itemlist')

#load presaved validation set
print('loading presaved valid set')                         
start_load = datetime.now()
valid = load_data(data_path,presaved_validset)
end_load = datetime.now()
print('...took',str(end_load-start_load),'to load presaved valid set')
#add the training dataloader to the saved databunch
print('attaching to data')
data.valid_dl = train.valid_dl
#and now your full databunch is in the variable 'data'
print('Done! Your databunch contains',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
#make sure device is where it should be
data.device = device
print('youre on device:',data.device)
print('databunch preview:')
print(data)
print('saving')
