#this script was adapted from train_GTDB_read_LM_1round to test the training time on the new validset
#it's up from 6hrs to 13hrs - is that because validset is bigger or because distributed training isn't working when the databunch is loaded from a saved file for some reason?
#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
import gc

from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0) 

from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_noload')
encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_enc_noload')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/train_logs/log_GTDB_read_LM_noload')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/')
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/lrplot_GTDB_read_LM_noload.png'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')

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
lrate=5e-3 

max_seqs=1000 #this will total about 25M seqs in databunch, 2.5M in valid rest in train; going to need to do the shuffled batches of files or something to do more than this!
val_max_seqs = None #think this will be 80M seqs or so?
num_cycles=1

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

print('creating databunch')
skiprows = 0
val_skiprows = 0
total_rounds = 0

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

config = awd_lstm_lm_config.copy()
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['bidir'] = False
config['emb_sz'] = emb_sz
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False).to_fp16()
#learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
learn.callbacks.append(TerminateOnNaNCallback()) 
#print(config)

if total_rounds==0: #get lr first round only
    print('figuring out LR')
    learn.lr_find()
    lr_plot = learn.recorder.plot(return_fig=True)
    lr_plot.savefig(lr_plot_out)

if total_rounds > 0:
    print('loading pretrained model')
    learn.load(model_path)

print('training cycles')
learn = learn.to_my_distributed(args.local_rank)
learn.fit_one_cycle(num_cycles,lrate, wd=wd, moms=moms)

print('saving model')
learn.save_encoder(encoder_path)
learn.save(model_path)

print('Done!')