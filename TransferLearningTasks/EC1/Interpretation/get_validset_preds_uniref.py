#add BioDL package to my path
import sys
sys.path.insert(0, '/home/ah1114/BioDL')
#and import all the stuff
from data import *
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_cpus",type=int)
args = parser.parse_args()

bs=8192 
ksize=1
n_cpus = int(args.n_cpus)
outfile = '/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/validset_predictions.csv'

print('getting predictions for validset')

#learn = load_learner('/home/ah1114/LanguageOfLife/TransferLearningTasks/EC1/models/', 'ec1_clas_best_export.pkl')
learn = load_learner('/home/ah1114/LanguageOfLife/saved_models', 'ec1_clas_uniref_best_export.pkl')
learn.data.batch_size = bs

valid_path = Path('/scratch/ah1114/LoL/TransferLearningTasks/EC1/valid/ec1_valid.csv')
valid_df = pd.read_csv(valid_path)
val_ds = BioTextList.from_df(valid_df,cols='seq')

learn.data.add_test(val_ds)

start = datetime.now()
preds,y = learn.get_preds(ds_type=DatasetType.Test)
end = datetime.now()
print('...took',str(end-start),'to get predictions')

class_ix = preds.argmax(dim=-1)
classes = [learn.data.classes[x] for x in class_ix]

pred_df = pd.DataFrame({'preds':preds,'predicted_label':classes,'seq':val_ds.items}) 
pred_df['pred_nonec1'] = [float(x[1]) for x in pred_df['preds']]
pred_df['pred_ec1'] = [float(x[0]) for x in pred_df['preds']]
pred_df.to_csv(outfile, index=False)
