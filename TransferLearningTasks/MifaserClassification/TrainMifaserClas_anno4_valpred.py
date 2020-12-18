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
import glob

model_path = Path('/home/ah1114/LanguageOfLife/saved_models/')
pretrained_path = 'mifaserclas_anno4_round11_export.pkl'

vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/')
train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/train/')
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/valid/')
#test_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/test/')

#if stuck with 128GB, might try doing it in shuffled batches of files?
bs=512 
n_cpus = args.n_cpus
ksize=1
stride=1
max_seqs=10
device = torch.device('cuda')

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

#do multiple rounds to avoid bus error 
num_rounds = 30
total_valrows = 2706868
step = int(total_valrows/num_rounds)

learn = load_learner(model_path, pretrained_path) 

for rnd in range(0,num_rounds):
    print('getting preds for rnd',rnd,'out of',num_rounds)
    skiprows = int(rnd*step)
    val_max_seqs = step

    start_bunch = datetime.now()
    data = BioClasDataBunch.from_multiple_csv(path=data_path, train=train_path, valid=valid_path,
                                        text_cols='seq', label_cols='annotation',
                                        classes=np.load('mifaser_classes.npy').tolist(),
                                        tokenizer=tok, vocab=model_voc,
                                        max_seqs_per_file=max_seqs, valid_max_seqs=val_max_seqs, skiprows=skiprows, bs=bs
                                            )
    end_bunch = datetime.now()
    print('...took',str(end_bunch-start_bunch),'to preprocess data')
    data.device = device
    print('youre on device:',data.device)
    data.num_workers = n_cpus 
    data.bs = bs
    data.batch_size = bs

    print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')
    print('there are',data.c,'classes:',data.classes[0:10])

    learn.data = data

    print('getting predictions')
    start_pred = datetime.now()
    preds, targets = learn.get_preds(ordered=True)
    print('shape of returned predictions:',preds.shape)
    print('min and max value of prediction layer:',preds.min(),preds.max())
    predictions = np.argmax(abs(preds), axis = 1) #label chosen
    #scores = preds[np.arange(len(preds)), predictions] #actual score at that spot
    end_pred = datetime.now()
    print('...took',str(end_pred-start_pred),'to get predictions')
    pred_names = [data.classes[x] for x in predictions]
    true_names = [data.classes[y] for y in targets]

    preds_out = '/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/MifaserClasPredictions_valid/MifaserClasPredictions_valid'+str(rnd)+'.pkl'
    print('saving predictions and targets to:',preds_out)
    preddf = pd.DataFrame({'predictions':predictions,'true':targets,'pred_names':pred_names,'true_names':true_names,'preds':preds.tolist()})
    print(preddf.head())
    print('dim preds items:',len(preddf['preds'][0]))
    print('preddf info:')
    print(preddf.info())
    preddf.to_pickle(preds_out)

#concatenate all the rounds into one file
files = glob.glob('/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/MifaserClasPredictions_valid/*.pkl')
df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
df.to_pickle('/scratch/ah1114/LoL/TransferLearningTasks/MifaserFn/MifaserClasPredictions_valid.pkl')

print('Done!')
