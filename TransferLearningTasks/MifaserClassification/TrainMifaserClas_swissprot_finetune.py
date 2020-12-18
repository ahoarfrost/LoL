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
model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_swissprot_best')
last_model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_swissprot')
model_dir = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/')
cache_dir = 'tmp_mifaserclas_swissprot'
#last_encoder_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_last_enc_round')
pretrained_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_round11')
log_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/train_logs/mifaserclas_swissprot')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/test/')
data_fname = 'swissprot_test.csv'
lr_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lrplot_mifaserclas_swissprot.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lossesplot_mifaserclas_swissprot.png'


#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=512 
ksize=1 
stride=1
lrate=1e-3 #adjusted down after seeing lrplot
drop_mult=0.3 #haven't tuned this at all yet
device = torch.device('cuda')
n_layers=3
n_hid=1152
#wd=1e-2 #using default wd for classification; may want to revisit
moms=(0.8,0.7) #lowered this for classification; may want to revisit
emb_sz=104
n_cpus = args.n_cpus

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)


start_bunch = datetime.now()
data = BioClasDataBunch.from_csv(path=data_path, csv_name=data_fname, valid_pct=0.15,
                                    text_cols='seq', label_cols='annotation',
                                    classes=np.load('mifaser_classes.npy').tolist(),
                                    tokenizer=tok, vocab=model_voc, bs=bs
                                        )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
data.device = device
print('youre on device:',data.device)
data.num_workers = n_cpus 
data.bs = bs

print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('there are',data.c,'classes')

config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification!
config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['emb_sz'] = emb_sz
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False)
learn.to_fp16()
#learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path)) #save best model if accuracy is above the best seen so far
learn.callbacks.append(SaveModelCallback(learn, every='epoch', monitor='accuracy', name=last_model_path)) #save latest model at end every epoch
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))

#this is where you would load the pretrained encoder learn.load_encoder('<ENC NAME>')
learn.load(pretrained_path)

print('figuring out LR')
start_lr = datetime.now()
learn.lr_find()
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)
end_lr = datetime.now()
print('...took',str(end_lr-start_lr),'to get lr')

print('training cycles')

#round0
learn.freeze()
learn.fit_one_cycle(10,5e-2, moms=(0.8,0.7))
#round1
learn.freeze_to(-2)
learn.fit_one_cycle(10, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7)) #this 2.6 rule of thumb is for NLP, maybe doesn't hold... but we'll go with it for now
#round2
learn.freeze_to(-3)
learn.fit_one_cycle(10, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7)) 
#round3
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7)) 

print('Done!')
