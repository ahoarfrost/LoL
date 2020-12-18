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
model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_round')
model_dir = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/')
cache_dir = 'tmp_mifaserclas4_round'
#last_encoder_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_last_enc_round')
pretrained_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_round11')
log_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/train_logs/mifaserclas_anno4_round')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/')
train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/train/')
#valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/valid/')
test_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/test/')
lr_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lrplot_mifaserclas_anno4_test.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lossesplot_mifaserclas_anno4_test.png'
preds_out = '/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/MifaserClasPredictions.csv'


#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=512 
ksize=1 
stride=1
max_seqs=10 
test_max_seqs=None
lrate=1e-3 #adjusted down after seeing lrplot
drop_mult=0.3 #haven't tuned this at all yet
num_cycles=1
skiprows_file = "skiprows_mifaser_anno4.csv"
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
data = BioClasDataBunch.from_multiple_csv(path=data_path, train=train_path, valid=test_path,
                                    text_cols='seq', label_cols='annotation',
                                    classes=np.load('mifaser_classes.npy').tolist(),
                                    tokenizer=tok, vocab=model_voc,
                                    max_seqs_per_file=max_seqs, valid_max_seqs=test_max_seqs, skiprows=0, bs=bs
                                        )
end_bunch = datetime.now()
print('...took',str(end_bunch-start_bunch),'to preprocess data')
data.device = device
print('youre on device:',data.device)
data.num_workers = n_cpus 
data.bs = bs

print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
print('there are',data.c,'classes:',data.classes[0:10])

config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification!
config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['emb_sz'] = emb_sz
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False)
#learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
#learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
#this is where you would load the pretrained encoder learn.load_encoder('<ENC NAME>')
learn.load(pretrained_path)

print('getting predictions')
start_pred = datetime.now()
preds, targets = learn.get_preds(ordered=True)
print('shape of returned predictions:',preds.shape)
print('min and max value of prediction layer:',preds.min(),preds.max())
predictions = np.argmax(abs(preds), axis = 1) 
end_pred = datetime.now()
print('...took',str(end_pred-start_pred),'to get predictions')
pred_names = [data.classes[x] for x in predictions]
true_names = [data.classes[y] for y in targets]

print('saving predictions and targets to:',preds_out)
preddf = pd.DataFrame({'predictions':predictions,'true':targets,'pred_names':pred_names,'true_names':true_names})
preddf.to_csv(preds_out,index=False)

print('Done!')
