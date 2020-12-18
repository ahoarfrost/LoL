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
train_path = Path('/scratch/ah1114/LoL/data/testGenomestrain')
valid_path = Path('/scratch/ah1114/LoL/data/testGenomesvalid')
outpath = Path('/scratch/ah1114/LoL/data/')
model_dir = Path('/scratch/ah1114/LoL/data/tmpmini/')
model_path_base = 'minibunchlm'
data_outfile = 'GTDB_minidatabunch.pkl'
log_path = Path('/scratch/ah1114/LoL/data/minibunch_log')

#params from parameter search
bs=512 
ksize=1
stride=1
n_layers=3
n_hid=1152
drop_mult=0.1
wd=1e-2 
moms=(0.98,0.9)
emb_sz=104
bptt=100
n_cpus=28

max_seqs=None #this will total about 20-25M seqs in databunch
val_max_seqs=None #this is about 11M seqs

skiprows=0
val_skiprows=0

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

print('creating databunch')
start_bunch = datetime.now()
data = BioLMDataBunch.from_folder(path=outpath, 
                                        train=train_path, valid=valid_path, ksize=ksize,
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, val_maxseqs=val_max_seqs,
                                        skiprows=skiprows, val_skiprows=val_skiprows, bs=bs, bptt=bptt
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


#test training on it
#create new training chunk 
print('loading databunch')
start_bunch = datetime.now()
data = load_data(outpath,data_outfile, bs=bs,bptt=bptt) #YOU NEED TO SET YOUR BS/BPTT HERE OR THE LANGUAGEMODELPRELOADER BREAKS EVERYTHING. also dl_tfms,device,collate_fn,no_check, etc. if they're not default. see docs
#data.bs,data.batch_size = bs,bs
#data.bptt = bptt
end_bunch = datetime.now()
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
#make sure device is where it should be
#data.device = device
data.num_workers = n_cpus

config = awd_lstm_lm_config.copy()
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['bidir'] = False
config['emb_sz'] = emb_sz
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False) 
#note you can create a learner with 

def train_round(learn,rnd,num_cycles,lrate):
    print('training',num_cycles,'cycle/s with lrate',lrate)
    callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', name=Path(model_path_base+rnd+'_best')),
                SaveModelCallback(learn, every='epoch', monitor='accuracy', name=Path(model_path_base+rnd)),
                CSVLogger(learn, filename=log_path, append=True)]
    learn.fit_one_cycle(num_cycles,lrate, wd=wd, moms=moms, callbacks=callbacks) 
    print('saving checkpoint encoder')
    learn.save_encoder(Path(model_path_base+rnd+'_latest_enc')) 
    print('Done!')

learn = learn.load('/home/ah1114/LanguageOfLife/saved_models/GTDB_read_LM_lowlr_continue4_best')
#loading weights doesn't affect trainability

train_round(learn, rnd=str(1), num_cycles=15, lrate=5e-4) 
#522805 items * 135bp / 100bptt = 705786 fragments
#bs=512 was 1400 batches (705786/512=1378)
#bs=2048 was 350 batches (705786/2048=345)