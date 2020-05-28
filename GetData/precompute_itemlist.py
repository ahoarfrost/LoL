import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from distributed import *
from fastai.callbacks import *
from datetime import datetime
import gc
import pickle

path = Path('./') 
vocab_path = Path('/home/ah1114/LanguageOfLife/vocabs')
vocab_name = vocab_path/'ngs_vocab_k1_withspecial.npy'
data_path = Path('/scratch/ah1114/LoL/data/')
train_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_train')
#valid_path = Path('/scratch/ah1114/LoL/data/GTDB_chunked_valid')

device = torch.device('cuda')
ksize=1
stride=1

chunk = int(sys.argv[1]) #e.g. 0
max_seqs = int(sys.argv[2]) #e.g. 16000
skiprows = int(sys.argv[3]) #e.g. 0
fout = '/scratch/ah1114/LoL/data/precomputed_itemlist/'+str(chunk)+'_'+str(skiprows)+'_'+str(max_seqs)+'.pkl'

tok = BioTokenizer(ksize=ksize, stride=stride)
if vocab_name.is_file():
    voc = np.load(vocab_name)
    model_voc = BioVocab(voc)
else:
    model_voc = BioVocab.create_from_ksize(ksize=ksize)
    np.save(vocab_name, model_voc.itos)

#create new training chunk
t_processor = [OpenSeqFileProcessor(max_seqs=max_seqs, ksize=ksize, skiprows=skiprows)] + get_lol_processor(tokenizer=tok, vocab=model_voc)
start_itemlist = datetime.now()
print('creating training set chunk')
data = BioTextList.from_folder(path=train_path, vocab=model_voc, max_seqs_per_file=max_seqs, processor=t_processor)
data = data.process()
print(len(data.items))
end_itemlist = datetime.now()
print('took',end_itemlist-start_itemlist,'to create itemlist')
#save
pickle.dump(data, open( fout, "wb" ) )