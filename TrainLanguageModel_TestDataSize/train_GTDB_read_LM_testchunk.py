#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
import gc 
from fastai.utils.mem import GPUMemTrace
mtrace = GPUMemTrace()

from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0) 
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

from fastai.distributed import * 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--n_cpus",type=int)
parser.add_argument("--maxseq",type=int)
parser.add_argument("--chunk",type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_testchunk')
encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_enc_testchunk')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/train_logs/log_GTDB_read_LM_testchunk')
data_path = Path('/scratch/ah1114/LoL/data/')
presaved_validset = 'GTDBdatabunch_validonly.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/lrplot_GTDB_read_LM_testchunk.png'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_trainparsed')
#valid_path = Path('/pylon5/eaz3a2p/horcrux/Wonderland/GTDB_chunked_valid')
device = torch.device('cuda')
#params
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
lrate=8e-3 

n_cpus=int(args.n_cpus)
max_seqs=int(args.maxseq)
#val_max_seqs=None #this will be 11M seqs
num_cycles=1
chunk = int(args.chunk)
skiprows = chunk*max_seqs
print('training chunk',chunk,'for',num_cycles,'cycles using max_seqs',max_seqs,'and skiprows',skiprows)

tok = BioTokenizer(ksize=ksize, stride=stride, n_cpus=n_cpus)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

#create new training chunk
t_processor = [OpenSeqFileProcessor(max_seqs=max_seqs, ksize=ksize, skiprows=skiprows)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
start_bunch = datetime.now()
print('creating training set chunk')
train = BioTextList.from_folder(path=train_path, vocab=model_voc, max_seqs_per_file=max_seqs, skiprows=skiprows, processor=t_processor)
val = train[[]]
val.ignore_empty = True
train = train._split(data_path,train,val)
train = train.label_for_lm()
train = train.databunch(bs=bs,bptt=bptt)
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess training data')
print('there are',len(train.items),'items in training itemlist')
#load presaved validation set
print('loading presaved valid set')                         
start_load = datetime.now()
data = load_data(data_path,presaved_validset)
end_load = datetime.now()
print('...took',str(end_load-start_load),'to load presaved valid set')
#add the training dataloader to the saved databunch
print('attaching to data')
data.train_dl = train.train_dl
print('bs and bptt:',bs,bptt)
#and now your full databunch is in the variable 'data'
print('Done! Your databunch contains',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
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
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False) #removed to_fp16() because getting nan losses on v100 cards...
#learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
#print(config)

if chunk==0: #get lr first round only
    print('figuring out LR')
    learn.lr_find()
    lr_plot = learn.recorder.plot(return_fig=True)
    lr_plot.savefig(lr_plot_out)
elif chunk>0:
    #load presaved model
    learn.load(model_path)

print('training',num_cycles, 'cycle/s')
learn = learn.to_my_distributed(args.local_rank)
learn.fit_one_cycle(num_cycles,lrate, wd=wd, moms=moms)

print('saving model')
learn.save_encoder(encoder_path)
learn.save(model_path)

print('Done!')