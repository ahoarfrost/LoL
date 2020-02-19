#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/LanguageOfLife/BioDL')
#and import all the stuff
from data import *
from fastai.callbacks import *
from datetime import datetime 

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/FrameClassification/models/frameclas_k1')
pretrained_path = Path('/home/ah1114/LanguageOfLife/FrameClassification/models/frameclas_k1')
log_path = Path('/home/ah1114/LanguageOfLife/FrameClassification/train_logs/frameclas_k1')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'

#data_path = Path('/scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv/')
data_path = Path('/scratch/ah1114/LoL/data/GTDBrepCDS_chunked_csv')

lr_plot_out = '/home/ah1114/LanguageOfLife/FrameClassification/lrplot_frameclas_k1.png'
#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=2048 #2048 worked in previous tests, 4096 had memory error
ksize=1
stride=1
max_seqs=100 
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

for round in range(0,8):
    print('creating databunch for round', round)
    skiprows_file = "skiprows.csv"
    skiprows = int(pd.read_csv(skiprows_file)['rows_to_skip']) #this should start at 0 at first training, will be updated as train more
    newskip = max_seqs+skiprows

    start_bunch = datetime.now()
    data = BioClasDataBunch.from_multiple_csv(path=data_path, 
                                        text_cols='seq', label_cols='frame',
                                        valid_pct=0.2, 
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, skiprows=skiprows, bs=bs
                                            )
    end_bunch = datetime.now()
    print('...took',str(end_bunch-start_bunch),'to preprocess data')

    print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
    print('there are',data.c,'classes')

    config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification!
    config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
    learn = text_classifier_learner(data, AWD_LSTM, drop_mult=drop_mult, model_dir=".", config=config, pretrained=False)
    learn.to_fp16()
    learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name=model_path))
    learn.callbacks.append(CSVLogger(learn, filename=log_path, append=True))
    #this is where you would load the pretrained encoder learn.load_encoder('<ENC NAME>')
    learn.load(pretrained_path)

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
