#add BioDL package to my path
import sys
sys.path.insert(0, '/home/horcrux/BioDL/')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
import gc 
from fastai.utils.mem import GPUMemTrace
mtrace = GPUMemTrace()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#from pynvml import *
#nvmlInit()
#handle = nvmlDeviceGetHandleByIndex(0) 
#info = nvmlDeviceGetMemoryInfo(handle)

#from fastai.distributed import * 
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--local_rank", type=int)
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()
#torch.cuda.set_device(args.local_rank)
#torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
model_path_base = '/home/horcrux/TrainLM_fastlr/models/train_LM_round'
model_dir = Path('/home/horcrux/TrainLM_fastlr/models/lm/')
log_path = Path('/home/horcrux/TrainLM_fastlr/train_logs/log_train_LM')
data_path = Path('/home/horcrux/data')
presaved_databunch = 'GTDB_class_databunch.pkl'
vocab_path = Path('/home/horcrux/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
lr_plot_base = '/home/horcrux/TrainLM_fastlr/plots/lrplot_train_LM_round'
losses_plot_base = '/home/horcrux/TrainLM_fastlr/plots/lossesplot_train_LM_round'
plot_lr_base = '/home/horcrux/TrainLM_fastlr/plots/plotlr_train_LM_round'
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
data = load_data(data_path,presaved_databunch, bs=bs, bptt=bptt)
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

#make distributed
#learn = learn.to_my_distributed(args.local_rank)

def train_round(learn,rnd,num_cycles,lrate):
    learn.lr_find(wd=wd, start_lr=1e-07, end_lr=1e-01) 
    lr_plot = learn.recorder.plot(return_fig=True)
    lr_plot.savefig(lr_plot_base+rnd+'.png')
    print('training',num_cycles,'cycle/s with lrate',lrate)
    callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', name=Path(model_path_base+rnd+'_best')),
                SaveModelCallback(learn, every='epoch', monitor='accuracy', name=Path(model_path_base+rnd)),
                CSVLogger(learn, filename=log_path, append=True)]
    learn.fit_one_cycle(num_cycles,lrate, wd=wd, moms=moms, callbacks=callbacks) 
    print('saving checkpoint encoder')
    learn.save_encoder(Path(model_path_base+rnd+'_latest_enc')) 
    print('saving recorder plots')
    plot_losses = learn.recorder.plot_losses(return_fig=True)
    plot_losses.savefig(losses_plot_base+rnd+'.png')
    print('Done!')


#train round 2 of 15 cycles with lrate 5e-4
#load presaved model
learn.load(model_path_base+'1_14')
train_round(learn, rnd=str(2), num_cycles=45, lrate=1e-3)
