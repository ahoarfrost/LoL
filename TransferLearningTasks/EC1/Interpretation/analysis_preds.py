import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

'''
chose metagenomes from TARA project for two stations in Pacific Ocean:
station 102 - upwelling on peru margin - ERR598962 (dcm) and ERR599055 (meso)
station 112 - SPG - ERR598957 (dcm) and ERR599072 (meso)
'''
metagenomes = ['ERR598962','ERR599055','ERR598957','ERR599072']
#metagenome_id = 'ERR598962'

for metagenome_id in metagenomes:
    print('analyzing metagenome',metagenome_id,'...')
    preds = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/'+str(metagenome_id)+'_cut_2M_predictions.csv')

    annotated = preds.dropna(subset=['mifaser_ec_1'])
    print('percent reads annotated:',len(annotated)/len(preds))
    true_ec1s = annotated[annotated['mifaser_ec_1'].str.startswith('1.')]
    true_nonec1s = annotated[~annotated['mifaser_ec_1'].str.startswith('1.')]
    annotated_predec1 = annotated[annotated['predicted_label']=='ec1']
    print('percent of predicted ec1 (with annotations) annotated as ec1 by mifaser:',len(annotated_predec1[annotated_predec1['mifaser_ec_1'].str.startswith('1.')])/len(annotated_predec1))
    annotated_prednonec1 = annotated[annotated['predicted_label']=='nonec1']

    agree = annotated_predec1[annotated_predec1['mifaser_ec_1'].str.startswith('1.')]
    print('average probability for agreed-upon ec1 predictions between LookingGlass and mifaser:',np.mean(agree['pred_ec1']))
    print('overall mean probability for LookingGlass ec1 predictions:',np.mean(preds[preds['predicted_label']=='ec1']['pred_ec1']) )

    above_99 = annotated[annotated['pred_ec1']>0.99]  
    print('of annotated reads with LG probs > 0.99, how many also ec1 by mifaser?',len(above_99[above_99['mifaser_ec_1'].str.startswith('1.')])/len(above_99))

    def get_scores(labels, preds, cutoff):
        predicted = preds>=cutoff
        true = labels.str.startswith('1.')
        
        TP = np.sum(np.logical_and(predicted == True, true == True))
        TN = np.sum(np.logical_and(predicted == False, true == False))
        FP = np.sum(np.logical_and(predicted == True, true == False))
        FN = np.sum(np.logical_and(predicted == False, true == True))
        
        acc = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*((precision*recall)/(precision+recall))
        
        return acc,precision,recall,f1

    metrics = pd.DataFrame()
    for cutoff in [0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0]:
        acc,precision,recall,f1 = get_scores(labels=annotated['mifaser_ec_1'], preds=annotated['pred_ec1'], cutoff=cutoff)
        metrics = metrics.append([[cutoff,acc,precision,recall,f1]])
    metrics.columns = ['cutoff','accuracy','precision','recall','f1']

    metrics.to_csv('/scratch/ah1114/LoL/TransferLearningTasks/EC1/metagenomes/'+str(metagenome_id)+'_cut_2M_metrics.csv', index=False)

    colors = sns.color_palette('Set1',4)
    fig,ax = plt.subplots()
    plt.plot( 'cutoff', 'accuracy', data=metrics, color=colors[0], linewidth=3)
    plt.plot( 'cutoff', 'precision', data=metrics, color=colors[1], linewidth=3)
    plt.plot( 'cutoff', 'recall', data=metrics, color=colors[2], linewidth=3)
    plt.plot( 'cutoff', 'f1', data=metrics, color=colors[3], linewidth=3)
    plt.legend()
    ax.set_xticks(ticks=np.arange(0.5,1.0,0.01),minor=True)
    ax.set_yticks(ticks=np.arange(0,1.0,0.1),minor=True)
    ax.set_xlabel('prediction probability threshold')
    plt.grid(b=True,which='minor',color='lightgray')
    plt.grid(b=True,which='major',color='lightgray')
    plt.savefig('/home/ah1114/LanguageOfLife/TransferLearningTasks/EC1/Interpretation/'+metagenome_id+'_metrics.png',dpi=150)
