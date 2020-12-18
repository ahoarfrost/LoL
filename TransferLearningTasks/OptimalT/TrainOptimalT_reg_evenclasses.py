#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime
import gc 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()

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

path = Path('./') 
data_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/OptimalT/')
train_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/OptimalT/temp_train_cats.csv')
valid_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/OptimalT/temp_valid_cats.csv')
model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/OptimalT/models/temp_reg_best')
last_model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/OptimalT/models/temp_reg')
log_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/OptimalT/train_logs/temp_reg_log')
lr_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/OptimalT/plots/lrplot_temp_reg.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/OptimalT/plots/lossesplot_temp_reg.png'
pretrained_path = Path('/home/ah1114/LanguageOfLife/saved_models/LookingGlass_enc')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'  
cache_dir = 'tmp_temp' #otherwise your tmp recorder stuff will all go in the same place if you're running multiple jobs
device = torch.device('cuda')

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
#data = BioClasDataBunch.from_df(path=data_path,train_df=train_df,valid_df=valid_df,
#                tokenizer=tok, vocab=model_voc, classes=['temp'],
#                text_cols='seq',label_cols='temp',
#                bs=bs,device=device
#                )

#data.batch_size = bs

#create databunch for regression
print('getting processor')
processor = get_lol_processor(tokenizer=tok, vocab=model_voc)
print('creating itemlists')
src = ItemLists(data_path, BioTextList.from_df(train_df, data_path, cols='seq', processor=processor),
                        BioTextList.from_df(valid_df, data_path, cols='seq', processor=processor))
print('labeling')
src = src.label_for_df_for_regression(cols='temp', classes=['temp'], label_cls=FloatList)
print('creating databunch')
data = src.databunch()
data.batch_size = bs

end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('there are',data.c,'classes')
#make sure device is where it should be
data.device = device
print('youre on device:',data.device)
data.num_workers = n_cpus 

def rmse(preds, targs):
    '''Compute root mean squared error'''
    return np.sqrt(torch.mean((targs - preds).pow(2)))

config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification/regression! (works for regression too)
config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['emb_sz'] = emb_sz
print('model config:')
print(config)
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False)
learn.loss_func = torch.nn.MSELoss()
learn.metrics = [rmse, root_mean_squared_error, R2Score(), ExplainedVariance()]
learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='rmse', mode='min', name=model_path)) #save best model if accuracy is above the best seen so far
learn.callbacks.append(SaveModelCallback(learn, every='epoch', monitor='rmse', mode='min', name=last_model_path)) #save latest model at end every epoch
learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
print('model:')
print(learn.model)

learn.load_encoder(pretrained_path)
#learn.load(pretrained_path)

print('checking out LR')
learn.lr_find()
lr_plot = learn.recorder.plot(return_fig=True)
lr_plot.savefig(lr_plot_out)

print('training cycle/s')
#learn = learn.to_my_distributed(args.local_rank)
#fine tune successive layers with discriminative learning rates
learn.freeze()
learn.fit_one_cycle(5,5e-2, moms=(0.8,0.7))
#learn.fit_one_cycle(10,1e-2, moms=(0.8,0.7))
learn.freeze_to(-2)
learn.fit_one_cycle(5, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7)) #this 2.6 rule of thumb is for NLP, maybe doesn't hold... but we'll go with it for now
#learn.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.freeze_to(-3)
learn.fit_one_cycle(5, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7)) 
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7)) 

print('plotting losses')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out)

print('Done!') 