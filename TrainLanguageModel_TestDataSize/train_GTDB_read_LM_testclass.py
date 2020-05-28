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

from fastai.distributed import * 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_testclass')
encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_enc_testclass')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_testclass/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/train_logs/log_GTDB_read_LM_testclass')
data_path = Path('/scratch/ah1114/LoL/data/')
#presaved_validset = 'GTDBdatabunch_validonly.pkl'
presaved_databunch = 'GTDB_class_databunch.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plots/lrplot_GTDB_read_LM_testclass.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plots/lossesplot_GTDB_read_LM_testclass.png'
plot_lr_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plots/plotlr_GTDB_read_LM_testclass.png'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train_parsedclass/')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid_parsedclass/')
cache_dir = 'tmp_testclass' #otherwise your tmp recorder stuff will all go in the same place if you're running multiple jobs
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
emb_sz=104
bptt=100
lrate=8e-3 

n_cpus=int(args.n_cpus)
max_seqs=None
val_max_seqs=None 
skiprows = 0
val_skiprows = 0

tok = BioTokenizer(ksize=ksize, stride=stride, n_cpus=n_cpus)
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
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False) #removed to_fp16() because getting nan losses on v100 cards...
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
print('config:',config)

print('figuring out LR')
learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

#load presaved model
#learn.load(model_path)

print('training cycle/s')
learn = learn.to_my_distributed(args.local_rank)

#custom learning schedule for this data size
learn.fit_one_cycle(1,lrate, wd=wd, moms=moms)
plot_lr = learn.recorder.plot_lr(return_fig=True)
plot_lr.savefig(plot_lr_out+'_1.png')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out+'_1.png')

#train some more with lower learning rate
learn.fit_one_cycle(3,1e-3, wd=wd, moms=moms)
plot_lr = learn.recorder.plot_lr(return_fig=True)
plot_lr.savefig(plot_lr_out+'_2.png')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out+'_2.png')

learn.fit_one_cycle(4,5e-4, wd=wd, moms=moms)
plot_lr = learn.recorder.plot_lr(return_fig=True)
plot_lr.savefig(plot_lr_out+'_3.png')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out+'_3.png')

print('saving model')
learn.save_encoder(encoder_path) 
learn.save(model_path)

print('Done!')