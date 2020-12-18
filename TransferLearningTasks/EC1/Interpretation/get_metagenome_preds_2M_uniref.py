#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from datetime import datetime
from Bio import SeqIO

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_cpus",type=int)
parser.add_argument("--metagenome_id",type=str)
parser.add_argument("--max_seqs",type=str)
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

bs=8192 #2048 works on 28 cpu; 4096 works; 8192 works; 16384 works; 32768 bus error; 8192 and 16384 take equal time to run for same amt data (32000 seqs, ~1.5min); 4096 is only slightly longer; 2048 is significantly slower; 8192 seems to be the sweet spot
ksize=1
skiprows=0
n_cpus = int(args.n_cpus)
metagenome_id = str(args.metagenome_id)
max_seqs = int(args.max_seqs)

path = Path('./') 
data_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/')
metagenome_path = str(metagenome_id)+'_cut.fastq'
outfile = '/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/'+str(metagenome_id)+'_cut_2M_predictions.csv'

print('getting predictions for metagenome',metagenome_id,'...')

#learn = load_learner('/home/ah1114/LanguageOfLife/TransferLearningTasks/EC1/models/', 'ec1_clas_best_export.pkl')
learn = load_learner('/home/ah1114/LanguageOfLife/saved_models/', 'ec1_clas_uniref_best_export.pkl')
learn.data.batch_size = bs
learn.data.num_workers = n_cpus

test_data = BioTextList.from_seqfile(filename=data_path/metagenome_path, path=data_path,
                                        max_seqs_per_file=max_seqs, skiprows=skiprows, ksize=ksize)

print('getting predictions for',len(test_data.items),'seqs...')

learn.data.add_test(test_data)

start = datetime.now()
preds,y = learn.get_preds(ds_type=DatasetType.Test)
end = datetime.now()
print('...took',str(end-start),'to get predictions for',max_seqs,'seqs with batch size',bs)

class_ix = preds.argmax(dim=-1)
classes = [learn.data.classes[x] for x in class_ix]

pred_df = pd.DataFrame({'preds':preds,'predicted_label':classes,'seq':test_data.items}) 
pred_df.to_csv(outfile)

def get_seqheaders(fname,ftype='fastq',max_seqs=max_seqs, skiprows=skiprows):
    ix=0
    descriptions = []
    for record in SeqIO.parse(fname,ftype):
        ix += 1
        if ix<=(max_seqs+skiprows) and ix>skiprows:
            desc = record.description
            new_desc = ''.join(desc.split(' '))
            descriptions.append(new_desc)
        if max_seqs and (ix-skiprows)>=max_seqs: 
            break
    return descriptions

#link these predictions with the mifaser annotations
ec_count = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/mifaser_metagenomes/'+str(metagenome_id)+'_cut/ec_count.tsv',sep='\t', names=['desc','mifaser_ec_1','mifaser_ec_2']) 
descriptions = get_seqheaders(data_path/metagenome_path, ftype='fastq',max_seqs=max_seqs,skiprows=skiprows)
pred_df['desc'] = descriptions
joined = pred_df.merge(ec_count,on='desc',how='left')
joined['pred_nonec1'] = [float(x[1]) for x in joined['preds']]
joined['pred_ec1'] = [float(x[0]) for x in joined['preds']]
joined.to_csv(outfile)

'''
#tested that this worked with:
valid_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/EC1/valid/ec1_valid.csv')
valid_df = pd.read_csv(valid_path)
val_ds = BioTextList.from_df(valid_df.loc[0:2000],cols='seq')
learn.data.add_test(val_ds)
preds,y = learn.get_preds(ds_type=DatasetType.Test)
#note the y is just an array of 0s for test set; if labels weren't empty this would be the 'true' value
class_ix = preds.argmax(dim=-1) #predicted classes
classes = [learn.data.classes[x] for x in class_ix]
'''