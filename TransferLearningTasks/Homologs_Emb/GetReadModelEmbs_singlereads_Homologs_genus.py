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
data_inpath = Path('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG_seqs_genus/')
outfile = Path('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/LookingGlass_HomEmb_out/OG_SeqEmbs_genus.pkl')

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
    if ix % 1000 == 0:
        print('processing fasta',ix,'out of',len(contents))
    tax = fna.parts[-2]
    og = fna.stem
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
time_end = datetime.now()
print('took',time_end-time_start,'to process')
print('saving processed dataframe')
to_write.to_pickle(outfile)

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
