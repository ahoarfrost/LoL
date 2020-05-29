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
model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_best2')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLM/models/mifaserclas2/')
cache_dir = 'tmp_mifaserclas'
last_model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas2')
last_encoder_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_last_enc2')
pretrained_path = Path('/home/ah1114/LanguageOfLife/saved_models/GTDB_read_LM_lowlr_continue2_best_enc')
log_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/train_logs/mifaserclas2')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data')
train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data/mifaser_train.csv')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/cdhit_clean_for_training/data/mifaser_valid.csv')
lr_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lrplot_mifaserclas2.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lossesplot_mifaserclas2.png'

device = torch.device('cuda')
bs=512 
ksize=1
stride=1
n_layers=3
n_hid=1152
drop_mult=0.1
#wd=1e-2 #using default wd for classification; may want to revisit
moms=(0.8,0.7) #lowered this for classification; may want to revisit
emb_sz=104
lrate=1e-2
n_cpus = args.n_cpus
num_cycles = 300

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

start_bunch = datetime.now()
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
classes = list(set(train_df['annotation']))
data = BioClasDataBunch.from_df(path=data_path,train_df=train_df,valid_df=valid_df,
                tokenizer=tok, vocab=model_voc, classes=classes,
                text_cols='seq',label_cols='annotation',
                bs=bs,device=device
                )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('there are',data.c,'classes')
#make sure device is where it should be
data.device = device
print('youre on device:',data.device)
data.num_workers = n_cpus 

config = awd_lstm_clas_config.copy() 
config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['emb_sz'] = emb_sz
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False)
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path)) #save best model if accuracy is above the best seen so far
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))

learn.load_encoder(pretrained_path)
#learn.load(pretrained_path)

print('checking out LR')
learn.lr_find(start_lr=1e-6,end_lr=1)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

print('training cycle/s')
learn = learn.to_my_distributed(args.local_rank)
#fine tune all at once with constant learning rate 
learn.fit_one_cycle(num_cycles,lrate, moms=(0.8,0.7))

print('plotting losses')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out)

print('saving last cycle encoder')
learn.save_encoder(last_encoder_path)

print('Done!')