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
model_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_testorder')
encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_enc_testorder')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/train_logs/log_GTDB_read_LM_testorder')
data_path = Path('/scratch/ah1114/LoL/data/')
#presaved_validset = 'GTDBdatabunch_validonly.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plots/lrplot_GTDB_read_LM_testorder.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plots/lossesplot_GTDB_read_LM_testorder.png'
plot_lr_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plots/plotlr_GTDB_read_LM_testorder.png'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train_parsedorder/')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid_parsedorder/')
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
val_max_seqs=None #this will be 11M seqs
skiprows = 0
val_skiprows = 0

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

#train each cycle with decreasing lrs
#doing range 8e-3 to 5e-4 with logarithmic decrease
lrs = [0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005]
for ix,lr in enumerate(lrs):
    learn.fit(1,lr,wd=wd) #note there's no moms input here
    plot_lr = learn.recorder.plot_lr(return_fig=True)
    plot_lr.savefig(plot_lr_out+'_'+str(ix)+'.png')
    plot_losses = learn.recorder.plot_losses(return_fig=True)
    plot_losses.savefig(losses_plot_out+'_'+str(ix)+'.png')
    print('metrics:',learn.metrics,type(learn.metrics))
    print('saving model')
    learn.save_encoder(encoder_path) 
    learn.save(model_path)

print('Done!')