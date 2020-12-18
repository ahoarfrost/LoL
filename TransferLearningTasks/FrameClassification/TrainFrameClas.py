#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/models/frame_clas_best')
last_model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/models/frame_clas')
model_dir = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/models/')
cache_dir = 'tmp_frameclas'
#last_encoder_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/models/frame_clas_enc')
pretrained_path = Path('/home/ah1114/LanguageOfLife/saved_models/LookingGlass_enc')
log_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/train_logs/frame_clas_log')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv')
train_path = Path('/scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv_train_parsedorder/')
valid_path = Path('/scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv_valid_parsedorder/') 
lr_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/plots/lrplot_frame_clas.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/FrameClassification/plots/lossesplot_frame_clas.png'

bs=512 
ksize=1
stride=1
n_layers=3
n_hid=1152
drop_mult=0.3
#wd=1e-2 #using default wd for classification; may want to revisit
moms=(0.8,0.7) #lowered this for classification; may want to revisit
emb_sz=104
lrate=1e-2
n_cpus = args.n_cpus
device = torch.device('cuda')

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

start_bunch = datetime.now()
data = BioClasDataBunch.from_multiple_csv(path=data_path, train=train_path, valid=valid_path,
                                    text_cols='seq', label_cols='frame',
                                    tokenizer=tok, vocab=model_voc,
                                    max_seqs_per_file=None, valid_max_seqs=None, skiprows=0, bs=bs
                                        )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('there are',data.c,'classes')
#make sure device is where it should be
data.device = device
print('youre on device:',data.device)
data.num_workers = n_cpus 

config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification!
config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['emb_sz'] = emb_sz
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False)
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path)) #save best model if accuracy is above the best seen so far
learn.callbacks.append(SaveModelCallback(learn, every='epoch', monitor='accuracy', name=last_model_path)) #save latest model at end every epoch
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))

#this is where you load the pretrained encoder learn.load_encoder('<ENC NAME>')
learn.load_encoder(pretrained_path)

print('checking out LR')
learn.lr_find(start_lr=1e-6,end_lr=1)
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

print('training cycle/s')
#fine tune successive layers with discriminative learning rates
learn.freeze()
learn.fit_one_cycle(5,1e-3, moms=moms)
learn.freeze_to(-2)
learn.fit_one_cycle(10, slice(5e-4/(2.6**4),5e-4), moms=moms) #this 2.6 rule of thumb is for NLP, maybe doesn't hold... but we'll go with it for now
learn.freeze_to(-3)
learn.fit_one_cycle(10, slice(1e-4/(2.6**4),1e-4), moms=moms) 
learn.unfreeze()
learn.fit_one_cycle(5, slice(5e-5/(2.6**4),5e-5), moms=moms) 

print('Done!')
