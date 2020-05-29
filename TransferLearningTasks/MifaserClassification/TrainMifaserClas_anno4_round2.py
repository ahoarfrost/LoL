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
model_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_round2')
model_dir = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas4round/')
cache_dir = 'tmp_mifaserclas4_round'
last_encoder_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_last_enc_round2')
pretrained_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/models/mifaserclas_anno4_round')
log_path = Path('/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/train_logs/mifaserclas_anno4_round')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/')
train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/train/')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/valid/')
lr_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lrplot_mifaserclas_anno4_round2.png'
losses_plot_out = '/home/ah1114/LanguageOfLife/TransferLearningTasks/MifaserClassification/plots/lossesplot_mifaserclas_anno4_round2.png'


#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=2048 #2048 worked in previous tests
ksize=1 
stride=1
max_seqs=5000000 
valid_max_seqs=None
lrate=1e-3 #adjusted down after seeing lrplot
drop_mult=0.3 #haven't tuned this at all yet
num_cycles=1
numrounds=5
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

for round in range(0,numrounds):
    print('creating databunch for round', round)
    skiprows = int(pd.read_csv(skiprows_file)['rows_to_skip']) #this should start at 0 at first training, will be updated as train more
    newskip = max_seqs+skiprows

    start_bunch = datetime.now()
    data = BioClasDataBunch.from_multiple_csv(path=data_path, train=train_path, valid=valid_path,
                                        text_cols='seq', label_cols='annotation',
                                        classes=np.load('mifaser_classes.npy').tolist(),
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, valid_max_seqs=valid_max_seqs, skiprows=skiprows, bs=bs
                                            )
    end_bunch = datetime.now()
    print('...took',str(end_bunch-start_bunch),'to preprocess data')
    data.device = device
    print('youre on device:',data.device)
    data.num_workers = n_cpus 

    print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
    print('there are',data.c,'classes')

    config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification!
    config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
    config['n_layers'] = n_layers
    config['n_hid'] = n_hid
    config['emb_sz'] = emb_sz
    learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=model_dir, config=config, pretrained=False)
    learn.to_fp16()
    learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
    learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
    #this is where you would load the pretrained encoder learn.load_encoder('<ENC NAME>')
    if round==0:
        learn.load(pretrained_path)
    else:
        learn.load(model_path)

    if round==0: #get lr first round only
        print('figuring out LR')
        start_lr = datetime.now()
        learn.lr_find()
        lr_plot = learn.recorder.plot(return_fig=True)
        lr_plot.savefig(lr_plot_out)
        end_lr = datetime.now()
        print('...took',str(end_lr-start_lr),'to get lr')

    print('training cycles')
    #note random guess accuracy 1/6 = 0.167
    start_train = datetime.now()
    learn.fit_one_cycle(num_cycles,lrate, moms=(0.8,0.7))
    end_train = datetime.now()
    print('...took',str(end_train-start_train),'to fit',num_cycles, 'cycles')

    print('adjusting skiprows value')
    pd.DataFrame({"rows_to_skip":[newskip]}).to_csv(skiprows_file,index=False) 

    print('Done!')

print('saving last cycle models')
learn.save_encoder(last_encoder_path)
print('plotting losses')
plot_losses = learn.recorder.plot_losses(return_fig=True)
plot_losses.savefig(losses_plot_out)