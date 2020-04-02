#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
import gc

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
model_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_1500tr')
encoder_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/GTDB_read_LM_enc_1500tr')
model_dir = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/models/')
log_path = Path('/home/ah1114/LanguageOfLife/TrainLanguageModel/train_logs/log_GTDB_read_LM_1500tr')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/')
presaved_validset = 'GTDBdatabunch_validonly.pkl'
lr_plot_out = '/home/ah1114/LanguageOfLife/TrainLanguageModel/lrplot_GTDB_read_LM_1500tr.png'
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')
skiprows_file = "skiprows.csv"
val_skiprows_file = "val_skiprows.csv"
round_file = "total_rounds.csv"
final_model=False
model_export = 'GTDB_read_LM.pkl'
device = torch.device('cuda')

#params from parameter search
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

max_seqs=1500 #2000 gave error; 1066 works; 1500? 
val_max_seqs=None
num_cycles=1
numrounds=10 

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

for round in range(0,numrounds):
    print('creating databunch for round',round)
    skiprows = int(pd.read_csv(skiprows_file)['rows_to_skip']) #this should start at 0 at first training, will be updated as train more
    if max_seqs: #if none, just keep skiprows
        newskip = max_seqs+skiprows
    else:
        newskip = skiprows
    val_skiprows = int(pd.read_csv(val_skiprows_file)['rows_to_skip']) #this should start at 0 at first training, will be updated as train more
    if val_max_seqs: #if none, just keep val_skiprows
        newvalskip = val_max_seqs+val_skiprows
    else:
        newvalskip = val_skiprows
    total_rounds = int(pd.read_csv(round_file)['round'])

    #create new training chunk
    t_processor = [OpenSeqFileProcessor(max_seqs=max_seqs, ksize=ksize, skiprows=0)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
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

    config = awd_lstm_lm_config.copy()
    config['n_layers'] = n_layers
    config['n_hid'] = n_hid
    config['bidir'] = False
    config['emb_sz'] = emb_sz
    learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False).to_fp16()
    #learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
    learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
    #print(config)

    if total_rounds==0: #get lr first round only
        print('figuring out LR')
        learn.lr_find()
        lr_plot = learn.recorder.plot(return_fig=True)
        lr_plot.savefig(lr_plot_out)

    if total_rounds > 0:
        learn.load(model_path)

    print('training cycles')
    learn = learn.to_my_distributed(args.local_rank)
    learn.fit_one_cycle(num_cycles,lrate, wd=wd, moms=moms)

    print('saving model')
    learn.save_encoder(encoder_path)
    learn.save(model_path)
    if final_model==True:
        learn.export(data_path,model_export)

    print('adjusting skiprows and round value')
    total_rounds = total_rounds+1
    pd.DataFrame({"rows_to_skip":[newskip]}).to_csv(skiprows_file,index=False) 
    pd.DataFrame({"rows_to_skip":[newvalskip]}).to_csv(val_skiprows_file,index=False) 
    pd.DataFrame({"round":[total_rounds]}).to_csv(round_file,index=False) 


print('Done!')