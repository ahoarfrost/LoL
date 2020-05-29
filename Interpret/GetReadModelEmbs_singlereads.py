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
pretrained_path = Path('/home/ah1114/LanguageOfLife/saved_models/GTDB_read_LM_loadsingle')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'  
data_path = Path('/scratch/ah1114/LoL/data/')
dummy_databunch = 'GTDBdatabunch_fortesting.pkl'
inference_datain = Path('/home/ah1114/LanguageOfLife/Interpret/InterpretSmallSubsetSeqs.csv')
chunk = int(sys.argv[1])
print('chunk:',chunk)
inference_out = Path('/home/ah1114/LanguageOfLife/Interpret/InterpretSmallSubset_Embs'+str(chunk)+'.pkl')

n_layers=3
n_hid=1152
emb_sz=100
drop_mult=0.2

#encoder_path = Path('/home/ah1114/LanguageOfLife/saved_models/best_GTDB_k1_1000seq_enc')
#data_path = Path('/home/ah1114/LanguageOfLife/Interpret')

data = load_data(data_path,dummy_databunch) 

config = awd_lstm_lm_config.copy()
config['n_layers'] = n_layers
config['n_hid'] = n_hid
config['bidir'] = False
config['emb_sz'] = emb_sz
learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult,model_dir=model_path, config=config, pretrained=False)#.to_fp16()
learn = learn.load(pretrained_path)

#fn to get the tokenized/numericalized processed seq from raw input
def process_seq(seq,learn):
    xb, yb = learn.data.one_item(seq)
    return xb

def encode_seq(learn, seq):
    xb = process_seq(seq, learn)
    encoder = learn.model[0]
    encoder.reset()
    with torch.no_grad():
        out = encoder.eval()(xb) #outputs tuple or raw vs dropped out eval
        #out[0] - raw eval; this is a list of length n_layers in model
        #out[0][-1] - last layer output; this is a tensor of size (batch_size, seq_len, emb_sz)
        #out[0][-1][0] - since this for a single sequence, take 0th index; now have tensor of size (seq_len, emb_sz)
        #out[0][-1][0][-1] - take embedding for last token; this is a tensor of size emb_sz
        #finally convert to numpy array and return
    # Return final output, for last layer, on last token in sequence
    return out[0][-1][0][-1].detach().numpy()

#in tests, processing 1000 rows took 19min; (a little over 1s per row)
skiprows=10949*chunk
nrows=10949
print('reading data with skiprows',skiprows,'and nrows',nrows)
if skiprows==0:
    to_process = pd.read_csv(inference_datain,skiprows=skiprows,nrows=nrows) #to_process is 43794 rows, so will take about 13 hours to run on all
else:
    to_process = pd.read_csv(inference_datain,skiprows=skiprows,nrows=nrows,names=['run','seq','annotation','annotation_level3','metaseek_env_package']) #to_process is 43794 rows, so will take about 13 hours to run on all
if 'emb' not in to_process.columns:
    to_process['emb'] = None
time_start = datetime.now()
num_rows=None
for ix,row in to_process.iterrows():
    if ix % 1000 == 0:
        print('processing row',ix,'out of',len(to_process))
    #print('processing row',ix)
    out = encode_seq(learn,row['seq'])
    to_process.at[ix,'emb'] = out   
    if num_rows:
        if ix>=num_rows:
            break
time_end = datetime.now()
print('took',time_end-time_start,'to process',len(to_process),'seqs')
print('saving processed dataframe')
to_process.to_pickle(inference_out)
#df = pd.read_pickle(inference_out) #to read back in, use read_pickle
#to_process.to_csv(inference_out,index=False) #this didn't work, all the embs were saved as strings with a '...' in the middle... 

#compute cosine similarity manually
def cosine(a,b):
    dot = np.dot(a,b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma*normb)
    return cos

#functions defined to encode manually; don't need to do with get_preds off RNNEncLearner

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
        out = enc.eval()(input) #tuple of length 2; 0th is raw out, 1st is dropped out output (should be the same in eval mode)
        raw_out = out[0] #a list of length n_layers
        last_layer_out = raw_out[-1] #output of last layer; tensor of shape (batch_size, seq_len, emb_sz)
        last_token_out = last_layer_out[:,-1] #output of last token; tensor of shape (batch_size, emb_sz)
    return last_token_out[0].detach().numpy() #convert to 1d tensor for output (since batch_size is 1 for single read, use 0th index)

#todo: encode_batch - just need to figure out processing for multiple sequences, then just input tensor with size (batch_size, seq_len) into enc.eval()
#from last_token_out, do last_token_out.detach().tolist() - will result in list of length (batch_size), each element is emb_sz; can then add as column 
