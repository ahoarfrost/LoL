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
model_path = Path('/home/ah1114/LanguageOfLife/TrainLM/models/GTDB_read_LM_lowlr_continue5_best')
last_model_path = Path('/home/ah1114/LanguageOfLife/TrainLM/models/GTDB_read_LM_lowlr_continue5')
last_encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLM/models/GTDB_read_LM_enc_lowlr_continue5_last')
model_dir = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/lowlr_continue5/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainLM/train_logs/log_GTDB_read_LM_lowlr_continue5')
data_path = Path('/scratch/ah1114/LoL/data/')
presaved_databunch = 'GTDB_class_databunch.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLM/plots/lrplot_GTDB_read_LM_lowlr_continue5.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TrainLM/plots/lossesplot_GTDB_read_LM_lowlr_continue5.png'
plot_lr_out = '/home/ah1114/LanguageOfLife/TrainLM/plots/plotlr_GTDB_read_LM_lowlr_continue5.png'
cache_dir = 'tmp_trainlm_lowlr_continue5' #otherwise your tmp recorder stuff will all go in the same place if you're running multiple jobs
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
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False) #removed to_fp16() because getting nan losses on v100 cards...
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path)) #save best model if accuracy is above the best seen so far
learn.callbacks.append(SaveModelCallback(learn, every='epoch', monitor='accuracy', name=last_model_path)) #save latest model at end every epoch
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
print('config:',config)

#load presaved model
learn.load(Path('/home/ah1114/LanguageOfLife/TrainLM/models/GTDB_read_LM_lowlr_continue4_best'))
#learn.load(last_model_path)

print('checking out LR')
learn.lr_find(wd=wd)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

print('training cycle/s')
learn = learn.to_my_distributed(args.local_rank)
#training 25 cycles, max will fit in my 3-day time limit
learn.fit_one_cycle(15,lrate, wd=wd, moms=moms)
#save last cycle model
print('saving last cycle models')
learn.save_encoder(last_encoder_path) 
learn.save(last_model_path)
#save plots
print('saving recorder plots')
#plot_lr = learn.recorder.plot_lr(return_fig=True)
#plot_lr.savefig(plot_lr_out)
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out)

print('Done!')