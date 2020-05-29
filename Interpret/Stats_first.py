import pandas as pandas

fsmall = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_test.csv'
fecoli = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretEColiPDB.csv'
fsmallfac = '/home/ah1114/LanguageOfLife/Interpret/InterpretSmallSubset_Embs.pkl'
small = pd.read_csv(fsmall) #names=['combo','seq_similarity','emb_similarity']
ecoli = pd.read_csv(fecoli,names=['combo','seq_similarity','emb_similarity','structural_similarity'])
smallfac = pd.read_pickle(fsmallfac)
#create separated df for manova stats - going to do in R because it's easier
exploded = pd.DataFrame(smallfac.emb.values.tolist(),index=smallfac.index) 
exploded['metaseek_env_package'] = smallfac['metaseek_env_package']  
exploded['annotation_level3'] = smallfac['annotation_level3']
exploded.to_csv('InterpretSmallSubset_Exploded.csv')
#need even group #s for manova - 

#correlate seq similarity and emb similarity - with both small and ecoli

#correlate structural similarity and emb similarity - with ecoli

#function manova - with smallfac
#env biome manova - with smallfac
#done in r - see manova output

fsmalla = '/scratch/ah1114/LoL/data/PairwiseSimilarities_InterpretSmallSubset_WithMismatchAlignlgth.csv'
smalla = pd.read_csv(fsmalla,nrows=1000000) #watch out this file is huge, use nrows

#note - next time I calculate these, calculate both cosine and euclidean emb distances, and record sequence length (so I can normalize to it)