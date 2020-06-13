import pandas as pd
import numpy as np

'''
chose metagenomes from TARA project for two stations in Pacific Ocean:
station 102 - upwelling on peru margin - ERR598962 (dcm) and ERR599055 (meso)
station 112 - SPG - ERR598957 (dcm) and ERR599072 (meso)
'''

metagenome_id = 'ERR598962'

preds = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/'+str(metagenome_id)+'_cut_predictions.csv')

annotated = joined.dropna(subset=['mifaser_ec_1'])
true_ec1s = annotated[annotated['mifaser_ec_1'].str.startswith('1.')]
true_nonec1s = annotated[~annotated['mifaser_ec_1'].str.startswith('1.')]