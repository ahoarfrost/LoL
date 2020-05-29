#get the sequence embeddings for all ENA sequence chunks - I ran this interactively since it's quick

import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from learner import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
from parse import *

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/saved_models/') 
pretrained_path = Path('/home/ah1114/LanguageOfLife/saved_models/GTDB_read_LM_loadsingle')
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'  
data_path = Path('/scratch/ah1114/LoL/data/')
dummy_databunch = 'GTDBdatabunch_fortesting.pkl'
inference_datain = Path('/scratch/ah1114/LoL/data/PDBtoENA_clean.csv')
ecoli_out = Path('/home/ah1114/LanguageOfLife/Interpret/InterpretEColiPDB_Embs.pkl')
bac_out = Path('/home/ah1114/LanguageOfLife/Interpret/InterpretBacPDB_Embs.pkl')

df = pd.read_csv(inference_datain)
#subset rows of inference_datain that correspond to E Coli (for which I have structural comparisons right now)
ecoli_ids = pd.read_csv('PDBforTopmatch_1chain50id2AEcoli.csv')
ecoli = pd.DataFrame()
for pid in ecoli_ids['PDB ID']:
    subset = df[df['PDBid']==pid]
    if len(subset)==0:
        print('missing Uniprot sequence for PDB id:',pid) #the Uniprot website has removed 4PTH=>B1XC49 because it is redundant (https://www.uniprot.org/uniprot/B1XC49)
    ecoli = ecoli.append(subset)
ecoli.reset_index(inplace=True,drop=True)  


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
    # Return final output, for last RNN, on last token in sequence
    return out[0][-1][0][-1].detach().numpy()

#in tests, processing 1000 rows took 19min; (a little over 1s per row)
#get embeddings for all the ecoli seqs
if 'emb' not in ecoli.columns:
    ecoli['emb'] = None
time_start = datetime.now()
for ix,row in ecoli.iterrows():
    if ix % 100 == 0:
        print('processing row',ix,'out of',len(ecoli))
    #print('processing row',ix)
    out = encode_seq(learn,row['seq'])
    ecoli.at[ix,'emb'] = out   
time_end = datetime.now()
print('took',time_end-time_start,'to process',len(ecoli),'seqs')
print('saving processed dataframe')
ecoli.to_pickle(ecoli_out)

#now do same for all bacteria 
if 'emb' not in df.columns:
    df['emb'] = None
time_start = datetime.now()
for ix,row in df.iterrows():
    if ix % 100 == 0:
        print('processing row',ix,'out of',len(df))
    #print('processing row',ix)
    out = encode_seq(learn,row['seq'])
    df.at[ix,'emb'] = out   
time_end = datetime.now()
print('took',time_end-time_start,'to process',len(df),'seqs')
print('saving processed dataframe')
df.to_pickle(bac_out)
