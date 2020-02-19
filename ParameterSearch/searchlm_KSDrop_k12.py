#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/LanguageOfLife/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime

from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

'''
from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
'''

path = Path('./') 
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')

data_path = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset/')
trainfolder = 'train'
validfolder = 'valid'

#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=8 #2048 worked in previous tests, 4096 had memory error
max_seqs=None
lrate=5e-3 
num_cycles=2

ksizes = [12,15]
strides = [1]
drop = [0.1,0.2,0.3,0.5]

for drop_mult in drop:
    for stride in strides:
        for ksize in ksizes:
            print('testing k=',ksize,', s=',stride,', dropout=',drop_mult)
            model_path = Path('/home/ah1114/LanguageOfLife/ParameterSearch/models/k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.')))
            model_name = Path('/home/ah1114/LanguageOfLife/ParameterSearch/models/k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.'))+'.pth')

            #check if model_path exists, continue if it doesn't
            #if model_name.is_file():
            #    print(model_name,'exists, skipping')
            #else:
            vocab_name = vocab_path/('ngs_vocab_k'+str(ksize)+'_withspecial.npy')
            log_path = Path('/home/ah1114/LanguageOfLife/ParameterSearch/train_logs/k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.')))
            lr_plot_out = '/home/ah1114/LanguageOfLife/ParameterSearch/lr_plots/lrplot_k'+str(ksize)+'_s'+str(stride)+'_drop'+''.join(str(drop_mult).split('.'))+'.png'

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
            info = nvmlDeviceGetMemoryInfo(handle)
            print("Total memory:", info.total)
            print("Free memory:", info.free)
            print("Used memory:", info.used)


            print('...there are',len(data.items),'items in train and',len(data.valid_ds.items),'items in valid')

            config = awd_lstm_lm_config.copy()
            config['bidir'] = False
            learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False).to_fp16()

            #learn = learn.to_distributed(args.local_rank)
            learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
            learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
            print('...model created')
            info = nvmlDeviceGetMemoryInfo(handle)
            print("Total memory:", info.total)
            print("Free memory:", info.free)
            print("Used memory:", info.used)

            learn.lr_find()
            lr_plot = learn.recorder.plot(return_fig=True)
            lr_plot.savefig(lr_plot_out)

            start_train = datetime.now()
            learn.fit_one_cycle(num_cycles,lrate, moms=(0.8,0.7))
            end_train = datetime.now()
            print('...took',str(end_train-start_train),'to fit',num_cycles, 'cycles')
            learn.save(model_path)

print('Done!')