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
best_model_path = Path('/home/ah1114/LanguageOfLife/TrainFinalLM/models/train_LM_best')
last_model_path = Path('/home/ah1114/LanguageOfLife/TrainFinalLM/models/train_LM_latest')
last_encoder_path = Path('/home/ah1114/LanguageOfLife/TrainFinalLM/models/train_LM_latest_enc')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainFinalLM/models/lm/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainFinalLM/train_logs/log_train_LM')
data_path = Path('/scratch/ah1114/LoL/data/')
presaved_databunch = 'GTDB_class_databunch.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lrplot_train_LM.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lossesplot_train_LM.png'
plot_lr_out = '/home/ah1114/LanguageOfLife/TrainFinalLM/plots/plotlr_train_LM.png'
cache_dir = 'tmp_trainlm' #otherwise your tmp recorder stuff will all go in the same place if you're running multiple jobs
device = torch.device('cuda')
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
#lrate=2e-3

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
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
#make sure device is where it should be
data.device = device
data.num_workers = n_cpus

config = awd_lstm_lm_config.copy()
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['bidir'] = False
config['emb_sz'] = emb_sz
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False) #removed to_fp16() because getting nan losses on v100 cards...
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=best_model_path))
learn.callbacks.append(SaveModelCallback(learn, every='epoch', monitor='accuracy', name=last_model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))

learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig('/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lrplot_train_LM0.png')

#load presaved model
#learn.load(best_model_path)

#first round of training with 2e-3 lr 
print('training 15 cycle/s with lrate 2e-3')
learn = learn.to_my_distributed(args.local_rank)
learn.fit_one_cycle(15,2e-3, wd=wd, moms=moms) #15 cycles reliably fits in 3-day time limit
print('saving checkpoint encoder')
learn.save_encoder(last_encoder_path) 
print('saving recorder plots')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig('/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lossesplot_train_LM0.png')

print('Done!')

#second round of training with lrate 5e-4
print('training four rounds of 15 cycle/s with lrate 5e-4')
learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig('/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lrplot_train_LM1.png')

learn.fit_one_cycle(15,5e-4, wd=wd, moms=moms)
print('saving checkpoint encoder')
learn.save_encoder(last_encoder_path) 

learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig('/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lrplot_train_LM2.png')

learn.fit_one_cycle(15,5e-4, wd=wd, moms=moms)
print('saving checkpoint encoder')
learn.save_encoder(last_encoder_path) 

learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig('/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lrplot_train_LM3.png')

learn.fit_one_cycle(15,5e-4, wd=wd, moms=moms)
print('saving checkpoint encoder')
learn.save_encoder(last_encoder_path) 

learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig('/home/ah1114/LanguageOfLife/TrainFinalLM/plots/lrplot_train_LM4.png')

learn.fit_one_cycle(15,5e-4, wd=wd, moms=moms)
print('saving checkpoint encoder')
learn.save_encoder(last_encoder_path) 

print('Done!')