#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime

from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0) 

from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--maxseq", type=int)
parser.add_argument("--skiprows",type=int)
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_datatest_500_2')
encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_enc_datatest_500_2')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_datatest_500/')
cache_dir = 'tmp_datatest500'
log_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/train_logs/log_GTDB_read_LM_datatest_500_2')
data_path = Path('/scratch/ah1114/LoL/data/')
presaved_validset = 'GTDBdatabunch_validonly.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/')
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/lrplot_GTDB_read_LM_datatest_500_2.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/lossesplot_GTDB_read_LM_datatest_500_2'
plot_lr_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/plotlr_GTDB_read_LM_datatest_500_2'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')
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
#use a range of max_seqs: 1, 10, 100, 500, 1000 - result in 20k, 200k, 2M, 10M, 20M in databunch respectively
max_seqs=int(args.maxseq) 
val_max_seqs=None #this will be 11M seqs
#num_cycles=10
skiprows = int(args.skiprows) #choose a random skiprows
print('using max_seqs',max_seqs,'and skiprows',skiprows)

tok = BioTokenizer(ksize=ksize, stride=stride)
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
train = BioTextList.from_folder(path=train_path, vocab=model_voc, max_seqs_per_file=max_seqs, processor=t_processor)
val = train[[]]
val.ignore_empty = True
train = train._split(data_path,train,val)
train = train.label_for_lm()
train = train.databunch()
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
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False)#.to_fp16()
#learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
#print(config)

#not loading any presaved stuff or doing lr_find
#if total_rounds==0: #get lr first round only
print('figuring out LR')
learn.lr_find()
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

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