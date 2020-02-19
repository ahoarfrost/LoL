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

ksize = 1
stride = 1

max_seqs=None
num_cycles=2
numrounds=1000

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

for round in range(0,numrounds):
    print('creating databunch for round', round,'out of',numrounds)
    #Adam beta2?
    np.random.seed(round)
    n_layers = int(np.random.choice([2,3,4]))
    np.random.seed(round)
    n_hid = int(np.random.choice([576,1152]))
    np.random.seed(round)
    drop_mult = float(np.random.choice([0.2,1.0]))
    np.random.seed(round)
    wd = float(np.random.choice([1e-1,1e-2,1e-3,1e-4]))
    mom_options = [(0.95,0.85),(0.9,0.8),(0.8,0.7)]
    np.random.seed(round)
    moms = mom_options[np.random.choice(len(mom_options))]
    np.random.seed(round)
    emb_sz = int(np.random.choice([50,100,125]))
    np.random.seed(round)
    bptt = int(np.random.choice([100,125,150]))
    np.random.seed(round)
    lrate = float(np.random.choice([2e-3,5e-3,8e-3]))
    np.random.seed(round)
    bs = int(np.random.choice([512,1024]))

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
    filename = Path('RandomSearchResults.csv')
    add_header = not filename.exists()
    if add_header: filename.open('a').write(','.join(params) + '\n')
    acc = float(learn.recorder.metrics[-1][0])
    stats = [ksize,stride,n_layers,n_hid,drop_mult,wd,moms,emb_sz,bptt,lrate,bs,acc]
    stats = [str(stat) for stat in stats]
    str_stats = ','.join(stats) 
    filename.open('a').write(str_stats+'\n')


print('Done!')