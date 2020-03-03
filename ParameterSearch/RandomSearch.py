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
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = Path('./') 
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')

data_path = Path('/scratch/ah1114/LoL/data/ModelSelectionMiniset/')
trainfolder = 'train'
validfolder = 'valid'


max_seqs=None
num_cycles=2
numrounds=1000
ksize=1
stride=1

vocab_name = vocab_path/('ngs_vocab_k'+str(ksize)+'_withspecial.npy')

tok = BioTokenizer(ksize=ksize, stride=stride)

if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

model_path = Path('/home/ah1114/LanguageOfLife/ParameterSearch/models/RandomSearch')
model_dir = Path('/home/ah1114/LanguageOfLife/ParameterSearch/models/')
log_path = Path('/home/ah1114/LanguageOfLife/ParameterSearch/train_logs/RandomSearch')
#lr_plot_out = '/home/ah1114/LanguageOfLife/ParameterSearch/lr_plots/RandomSearch.png'

with open('SearchParams.tsv') as f:  
    for line in f:  
        ksize,stride,n_layers,n_hid,drop_mult,wd,moms,emb_sz,bptt,lrate,bs = line.rstrip('\n').split('\t') 
        ksize=int(ksize)
        stride=int(stride)
        n_layers = int(n_layers)
        n_hid = int(n_hid)
        drop_mult = float(drop_mult)
        wd = float(wd)
        moms = tuple([float(x) for x in moms.split('(')[1].split(')')[0].split(', ')])
        emb_sz = int(emb_sz)
        bptt = int(bptt)
        lrate = float(lrate)
        bs = int(bs)

        print('ksize','stride','n_layers','n_hid','drop_mult','wd','moms','emb_sz','bptt','lrate','bs')
        print(ksize,stride,n_layers,n_hid,drop_mult,wd,moms,emb_sz,bptt,lrate,bs)

        start_bunch = datetime.now()
        data = BioLMDataBunch.from_folder(path=data_path, 
                                            train=trainfolder, valid=validfolder, ksize=ksize,
                                            tokenizer=tok, vocab=model_voc,
                                            max_seqs_per_file=max_seqs, bs=bs, bptt=bptt
                                                )
        end_bunch = datetime.now()

        config = awd_lstm_lm_config.copy()
        config['n_layers'] = n_layers
        config['n_hid'] = n_hid
        config['bidir'] = False
        config['emb_sz'] = emb_sz
        learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False).to_fp16()

        learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
        learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))

        learn = learn.to_my_distributed(args.local_rank)

        learn.fit_one_cycle(num_cycles,lrate, wd=wd, moms=moms)

        params=['ksize','stride','n_layers','n_hid','drop_mult','wd','moms','emb_sz','bptt','lrate','bs','accuracy']
        filename = Path('RandomSearchResults2.csv')
        add_header = not filename.exists()
        if add_header: filename.open('a').write(','.join(params) + '\n')
        acc = float(learn.recorder.metrics[-1][0])
        stats = [ksize,stride,n_layers,n_hid,drop_mult,wd,moms,emb_sz,bptt,lrate,bs,acc]
        stats = [str(stat) for stat in stats]
        str_stats = ','.join(stats) 
        filename.open('a').write(str_stats+'\n')


print('Done!')