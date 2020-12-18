import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from learner import *
from datetime import datetime
import pandas as pd
from Bio import SeqIO

path = Path('./') 
model_path = Path('/home/ah1114/LanguageOfLife/saved_models/')
pretrained_path = 'LookingGlass_LM_export.pkl'
#data_inpath = Path('/scratch/ah1114/LoL/InterpretLG/InterpretSubsetEvenEnv.csv')
outpath = Path('/scratch/ah1114/LoL/InterpretLG/Embs_MifaserValid.pkl')
exploded_outpath = Path('/scratch/ah1114/LoL/InterpretLG/Embs_MifaserValid_Exploded.pkl')

valid = pd.read_csv('/scratch/ah1114/LoL/data/mifaser_out/Small100_EvenEnvPackages/clean_for_training/cdhit_processed_anno4/valid/mifaser_valid.csv')
print('number of ec1 classes at level 4',len(set(valid['annotation'])))
env = pd.read_csv('TrainSmall95_EvenEnv.csv') 
env.index = env['run_ids_maxrun']  
df = valid.join(env[['metaseek_env_package']],on='run')  
df = df.dropna()

#load pretrained LM
bptt = 100
bs = 512
learn = load_learner(model_path, pretrained_path) 
learn.data.bs = bs
learn.data.batch_size = bs
learn.data.bptt = bptt

#encoding one seq at a time - if you do a batch, they get grouped into bptt-length batches (for lmdatabunch) or near-equal length batches (for clasdatabunch) - neither ideal. 
#Can revisit batches if need speedup later
#fn to get the tokenized/numericalized processed seq from raw input
def process_seq(seq,learn):
    xb, yb = learn.data.one_item(seq)
    return xb

#fn to get the overall sequence embedding from a single sequence
def encode_seq(learn, seq): 
    xb = process_seq(seq, learn)
    encoder = learn.model[0]
    encoder.reset()
    with torch.no_grad():
        out = encoder.eval()(xb) #outputs tuple of raw vs dropped out eval
        #out[0] - raw eval; this is a list of length n_layers in model
        #out[0][-1] - last layer output; this is a tensor of size (batch_size, seq_len, emb_sz)
        #out[0][-1][0] - since this for a single sequence, take 0th index; now have tensor of size (seq_len, emb_sz)
        #out[0][-1][0][-1] - take embedding for last token; this is a tensor of size emb_sz
        #finally convert to numpy array and return
    # Return final output, for last layer, on last token in sequence - overall 'sequence embedding'
    return out[0][-1][0][-1].detach().cpu().numpy()

time_start = datetime.now()
embs = []
#for each of these reads, extract the run, seq, and annotation, and the labels from the filename
for ix,seq in enumerate(df['seq']):
    if ix % 1000 == 0:
        print('processing row',ix,'out of',len(df))
    
    emb = encode_seq(learn,seq)
    embs.append(emb)
    
time_end = datetime.now()
print('took',time_end-time_start,'to process') 

df['emb'] = embs
print('saving to',outpath)
df.to_pickle(outpath)
print('Done!')

#explode and save exploded
exploded = pd.DataFrame(df.emb.tolist())
exploded['seq'] = df['seq']
exploded['annotation'] = df['annotation']
exploded['metaseek_env_package'] = df['metaseek_env_package']
exploded['run'] = df['run']
exploded.to_pickle(exploded_outpath)
