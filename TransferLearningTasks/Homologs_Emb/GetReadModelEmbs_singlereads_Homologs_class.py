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
data_inpath = Path('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG_seqs_class/')
all_outfile = Path('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_HomEmb_out/OG_SeqEmbs_class.pkl')
outpath = Path('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/HomEmb_csv_class/')

#get a list of all the files in the nested folders of data_inpath
subdir = [x for x in data_inpath.iterdir()]
contents = []
for taxa in subdir:
    for x in taxa.iterdir():
        contents.append(x)

#load pretrained LM
bptt = 100
bs = 512
learn = load_learner(model_path, pretrained_path) 
learn.data.bs = bs
learn.data.bptt = bptt

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
    return out[0][-1][0][-1].detach().numpy()


time_start = datetime.now()
to_write = pd.DataFrame()
#data_writer = csv.writer(outfile)
#data_writer.writerow(['taxa','OG','seq','emb'])
#for each of these reads, extract the run, seq, and annotation, and the labels from the filename
for ix,fna in enumerate(contents):
    to_write = pd.DataFrame()
    if ix % 1000 == 0:
        print('processing fasta',ix,'out of',len(contents))
    tax = fna.parts[-2]
    og = fna.stem
    outfile = str(outpath/tax/og)+'.csv' #oops this should have been .pkl
    #if outpath/tax doesn't exist, create it
    (outpath/tax).mkdir(parents=True, exist_ok=True)
    #check if outfile already exists; if it does, can skip
    if Path(outfile).exists():
        print('fna file exists, skipping:',fna)
    else:
        records = []
        for record in SeqIO.parse(fna,"fasta"): 
            seq = str(record.seq)
            emb = encode_seq(learn,seq)
            record = [tax,og,seq,emb]    
            records.append(record)
        #write rows with taxa, og, seq, and emb 
        to_write = to_write.append(records)
        #data_writer.writerows(records)
        to_write.columns = ['taxa','og','seq','emb']
        to_write.reset_index(drop=True,inplace=True)
        to_write.to_pickle(outfile)
time_end = datetime.now()
print('took',time_end-time_start,'to process') 

#concatenate all the files 
#get a list of all the pickle files in the outpath
print('concatenating separate pkl files to one')
subdir = [x for x in outpath.iterdir()]
all_pickles = []
for taxa in subdir:
    for x in taxa.iterdir():
        all_pickles.append(x)
#read each pickle file, concatenate them together and write as single output
alldf = pd.DataFrame()
for p in all_pickles:
    onedf = pd.read_pickle(p)
    alldf = alldf.append(onedf)
alldf.reset_index(drop=True,inplace=True)
print('saving to',all_outfile)
alldf.to_pickle(all_outfile)
print('Done!')
