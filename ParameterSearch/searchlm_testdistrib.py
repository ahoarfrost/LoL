#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/LanguageOfLife/BioDL')
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
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')

data_path = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset/')
trainfolder = 'train'
validfolder = 'valid'

#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=2048 #2048 worked in previous tests, 4096 had memory error
max_seqs=None
lrate=5e-3 
num_cycles=2
drop_mult=0.2
ksize = 1
stride = 1

model_path = Path('/home/ah1114/LanguageOfLife/ParameterSearch/models/distrib_k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.')))
model_name = Path('/home/ah1114/LanguageOfLife/ParameterSearch/models/distrib_k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.'))+'.pth')

vocab_name = vocab_path/('ngs_vocab_k'+str(ksize)+'_withspecial.npy')
log_path = Path('/home/ah1114/LanguageOfLife/ParameterSearch/train_logs/distrib_k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.')))
lr_plot_out = '/home/ah1114/LanguageOfLife/ParameterSearch/lr_plots/distrib_lrplot_k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.'))+'.png'

tok = BioTokenizer(ksize=ksize, stride=stride)

if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

print('creating databunch')
start_bunch = datetime.now()
data = BioLMDataBunch.from_folder(path=data_path, 
                                    train=trainfolder, valid=validfolder, ksize=ksize,
                                    tokenizer=tok, vocab=model_voc,
                                    max_seqs_per_file=max_seqs, bs=bs
                                        )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('batch size is',data.bs)
info = nvmlDeviceGetMemoryInfo(handle)

print('...there are',len(data.items),'items in train and',len(data.valid_ds.items),'items in valid')

config = awd_lstm_lm_config.copy()
config['bidir'] = False
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False).to_fp16()

learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
learn.load(model_path) #if pretrained already

#learn.lr_find()
#lr_plot = learn.recorder.plot(return_fig=True)
#lr_plot.savefig(lr_plot_out)

learn = learn.to_my_distributed(args.local_rank)

print('...model created')
print('memory load after creating model:')
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

start_train = datetime.now()
learn.fit_one_cycle(num_cycles,lrate, moms=(0.8,0.7))
end_train = datetime.now()
print('...took',str(end_train-start_train),'to fit',num_cycles, 'cycles')
learn.save(model_path)

print('Done!')