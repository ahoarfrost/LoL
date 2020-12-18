import pandas as pd
from pathlib import Path


files = [f for f in Path('/scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes').iterdir()]
results = [x for x in files if x.match('*_cut_20M_predictions.csv')]

rows = []
for result in results:
    print('processing', result)
    df = pd.read_csv(result, index_col=0)
    srr = result.name.split('_')[0]

    #percent ec1 predicted
    num_ec1 = len(df[df['predicted_label']=='ec1'])
    percent_ec1 = num_ec1/len(df)

    rows.append([srr,percent_ec1])

summary = pd.DataFrame(rows, columns = ['SRR','percent_ec1'])
summary.to_csv('EC1_predictions_summary.csv')