#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/LanguageOfLife/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/TaxClassification/models/best_GTDB_k1_1000seq')
encoder_path = Path('/home/ah1114/LanguageOfLife/TaxClassification/models/best_GTDB_k1_1000seq_enc')
log_path = Path('/home/ah1114/LanguageOfLife/TaxClassification/train_logs/log_GTDB_k1_1000seq')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'

data_path = Path('/scratch/ah1114/LoL/data/GTDBrepGenomes_chunked/')

lr_plot_out = '/home/ah1114/LanguageOfLife/TaxClassification/lrplot_gtdb_k1_1000seq.png'
#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=2048 #2048 worked in previous tests, 4096 had memory error
ksize=1
stride=1
max_seqs=1066 #this will total about 25M seqs in databunch; going to need to do the shuffled batches of files or something to do more than this!
lrate=1e-2 #judging from ANI 0.95 no bidir cutoff; this should maybe be higher
drop_mult=0.3 #haven't tuned this at all yet
num_cycles=1

'''
def extract_frame(title): 
    ...:     import re 
    ...:     frame = re.split('\[frame=', title)[1][:-1]  
    ...:     return int(frame) 
'''

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
                                     valid_pct=0.2,
                                      tokenizer=tok, vocab=model_voc,
                                      max_seqs_per_file=max_seqs, bs=bs
                                         )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')

print('there are',len(data.items),'items in itemlist')

config = awd_lstm_lm_config.copy()
config['bidir'] = False
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False)
learn.to_fp16()
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))

print('figuring out LR')
start_lr = datetime.now()
learn.lr_find()
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)
end_lr = datetime.now()
print('...took',str(end_lr-start_lr),'to get lr')

print('training cycles')
start_train = datetime.now()
learn.fit_one_cycle(num_cycles,lrate, moms=(0.8,0.7))
end_train = datetime.now()
print('...took',str(end_train-start_train),'to fit',num_cycles, 'cycles')
print('saving final encoder')
learn.save_encoder(encoder_path)

print('Done!')