import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from learner import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/saved_models/')
pretrained_path = 'best_GTDB_k1_1000seq_encmodel_export.pkl'
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'

train_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/train/mifaser_train.csv') 
valid_path = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/valid/mifaser_valid.csv') 
outfile_trn = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/mifaser_train_withemb.csv')
outfile_val = Path('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/mifaser_valid_withemb.csv')

encoder_path = Path('/home/ah1114/LanguageOfLife/saved_models/best_GTDB_k1_1000seq_enc')
data_path = Path('scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages')

#params from parameter searchs
bs=4096 
ksize=1
stride=1
emb_sz=400 #**change this to 104 for real script, but I'm using the previous pretrained model with 400 so dims need to match

skiprows_file = "skiprows_emb.csv"
max_seqs=None #May need change

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)


skiprows = int(pd.read_csv(skiprows_file)['rows_to_skip']) #this should start at 0 at first training, will be updated as train more

#get valid embeddings 
print('opening valid')
valid_df = pd.read_csv(valid_path, nrows=max_seqs, skiprows=range(1,skiprows+1))
print('allocating learner')
learnenc = load_learner(model_path,pretrained_path)
learnenc.data.add_test(valid_df['seq']) 
learnenc.data.batch_size = bs
print('getting encoder embeddings for',len(learnenc.data.test_ds.items),'sequences')
start = datetime.now()
predx,predy = learnenc.get_preds(ds_type=DatasetType.Test) 
end = datetime.now()
print('took',end-start,'to finish predictions; saving to file',outfile_val)
valid_df['emb'] = predx  
valid_df.to_csv(outfile_val)

#get train embeddings
print('opening train')
train_df = pd.read_csv(train_path, nrows=max_seqs, skiprows=range(1,skiprows+1))
print('allocating learner')
learnenc = load_learner(model_path,pretrained_path)
learnenc.data.add_test(train_df['seq']) 
learnenc.data.batch_size = bs
print('getting encoder embeddings for',len(learnenc.data.test_ds.items),'sequences')
start = datetime.now()
predx,predy = learnenc.get_preds(ds_type=DatasetType.Test) 
end = datetime.now()
print('took',end-start,'to finish predictions; saving to file',outfile_trn)
train_df['emb'] = predx  
train_df.to_csv(outfile_trn)

#functions defined to encode manually; don't need to do with get_preds off RNNEncLearner
def process_seq(seq, processor):
    item=seq
    for p in processor:
        item = p.process_one(item)
    return LongTensor([item])

def encode_all_seqs(learn):
    enc = learn.model[0]
    enc.reset()
    if learn.data.device==torch.device('cuda'):
        embs = Tensor().cuda() 
    else:
        embs = Tensor()
    targets = []

    for dt in [DatasetType.Train,DatasetType.Valid,DatasetType.Test]: 
        if learn.dl(dt): 
                for (input, target) in learn.dl(dt):
                    with torch.no_grad():
                        #do forward pass on the model, parse the output to get the output embedding of the last token in the sequence 
                        #out is tuple with first element raw output, second element with dropout applied (in .eval mode no dropout applied, should be the same but whatevs)
                        out = enc.eval()(input)
                        raw_out = out[0]
                        #raw output is the output matrices from each layer. In our case, len(raw_out) is 3, output of the last layer is raw_out[2]
                        last_layer_out = raw_out[-1]
                        #shape of last_layer_out is (batch size, input sequence length, output emb size (which is equal to the emb_sz for us))
                        #the encoding we want is the last token in our document, which has the context of the whole preceding sequence (see the first line of the for loop in fastai.text.learner.LanguageLearner.predict)
                        last_token_out = last_layer_out[:,-1] 
                        #shape of last_token_out will be (batch size, output emb size) - the output emb of each item in batch
                        embs = torch.cat((embs,last_token_out))
                        targets.append(target)

    return embs

def encode_one_seq(seq,enc,processor):
    input = process_seq(seq,processor)
    enc.reset()
    with torch.no_grad():
        out = enc.eval()(input)
        raw_out = out[0]
        last_layer_out = raw_out[-1]
        last_token_out = last_layer_out[:,-1] 
    return last_token_out[0]

'''
data = BioClasDataBunch.from_df(path=data_path, train_df=train_df, valid_df=valid_df,
                                        text_cols='seq', label_cols='run',
                                        classes=list(np.load('run_accession_classes_100EvenEnv.npy')),
                                        tokenizer=tok, vocab=model_voc,
                                        bs=bs
                                                )

print('there are',len(data.items),'items in itemlist, and',len(data.valid_ds.items),'items in data.valid_ds')

#learn = load_learner(model_path, pretrained_path) 
#enc = learn.model[0]
#learnenc = RNNEncLearner(data, enc)
#learn.load(encoder_path)

#a way of doing this one seq at a time
#these aren't exaaaaactly the same as the ones using get_preds, but are very close; I think because with get_preds they're padded at the beginning
processor = learn.data.processor
start = datetime.now()
embs = []
for ix,row in valid_df.iterrows():
    emb = encode_one_seq(row['seq'],enc,processor) 
    embs.append(np.array(emb))
#df['emb'] = embs
end = datetime.now()
print('...took',str(end-start),'process',len(valid_df),'seqs')

'''
