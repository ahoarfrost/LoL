#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime
import gc 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/saved_models/LookingGlass_LM')
encoder_out = Path('/home/ah1114/LanguageOfLife/saved_models/LookingGlass_enc')
export_lm = Path('/home/ah1114/LanguageOfLife/saved_models/LookingGlass_LM_export.pkl')
#export_enc = Path('/home/ah1114/LanguageOfLife/saved_models/LookingGlass_enc_export')

data_path = Path('/scratch/ah1114/LoL/data/')
presaved_databunch = 'GTDBdatabunch_fortesting.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'

device = torch.device('cpu')
#params
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
lrate=5e-4

n_cpus=int(args.n_cpus)

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
data = load_data(data_path,presaved_databunch)
data.bs = bs
data.bptt = bptt
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to load data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
#make sure device is where it should be
data.device = device
print('youre on device:',data.device)
data.num_workers = n_cpus
print('databunch preview:')
print(data)

config = awd_lstm_lm_config.copy()
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['bidir'] = False
config['emb_sz'] = emb_sz
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False) #removed to_fp16() because getting nan losses on v100 cards...
print('config:',config)

#load presaved model
learn.load(model_path)

#save in various forms
learn.save_encoder(encoder_out)
learn.export(export_lm)

print('Done!')