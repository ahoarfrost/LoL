import pandas as pd
import requests
from collections import Counter
import urllib.parse 
import urllib.request 
from pathlib import Path
'''
xrefs = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/odb10v1_gene_xrefs.tab',sep='\t',header=None)
xrefs.columns = ['orthoDB_geneID','externalID','externalDB']
og2g = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/odb10v1_OG2genes.tab',sep='\t',header=False)
og2g.columns = ['OGID','orthoDB_geneID']
uniprot = xrefs[xrefs['externalDB']=='UniProt']
merged = uniprot.merge(og2g,on='orthoDB_geneID')
merged['NCBI_taxID'] = [int(x.split('at')[-1]) for x in merged['OGID']] 
levels = pd.read_csv("/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/odb10v1_levels.tab",sep="\t",header=None)
levels.columns = ['NCBI_taxID','scientific_name','total_genes','total_OGs','total_species']
uniprottax = merged.merge(levels[['NCBI_taxID','scientific_name']],on='NCBI_taxID') 
uniprottax.to_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes_withtaxa.csv',index=False)
#going to look up the embl 

joinedtax = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes_withtaxa.csv',index_col=False)
joinedtax['externalID'] = joinedtax['externalID'].astype(str) 
OGtaxa = ['Sulfolobus','Streptomyces','Pseudomonas','Thermococcus','Bacillus','Methanosarcina','Acinetobacter','Mycobacterium','Lactobacillus','Enterococcus']

subset = joinedtax[joinedtax['scientific_name'].isin(OGtaxa)] 
#map the uniprot IDs we care about to EMBL/GenBank/DDBJ CDS sequences - the uniprot database mapper is good for this
num_chunks = 50 #have to do in chunks or api crashes
interval = int(len(subset)/num_chunks)+1
starts = [x*interval for x in range(0,num_chunks)]
stops = [x*interval for x in range(1,num_chunks+1)] 
url = 'https://www.uniprot.org/uploadlists/'
for ix in range(0,num_chunks):
    #query uniprot database mapper API, see: https://www.uniprot.org/help/api_idmapping
    print('querying ix',ix)
    query = ' '.join(subset['externalID'][starts[ix]:stops[ix]])
    params = { 
    'from': 'ACC+ID', #uniprot KB id
    'to': 'EMBL', #EMBL/GenBank/DDBJ CDS
    'format': 'tab', 
    'query': query
    } 
     
    data = urllib.parse.urlencode(params) 
    data = data.encode('utf-8') 
    req = urllib.request.Request(url, data) 
    with urllib.request.urlopen(req) as f: 
       response = f.read() 
    print('...writing to file')
    #this will write tsv file with columns "From" and "To" - from is uniprot; to is embl cds
    with open('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprot2cds_'+str(ix)+'.tsv','w') as f: 
        f.writelines(response.decode('utf-8')) 
    #print(response.decode('utf-8'))      

    #save uniprot IDs in chunks to enter in online mapper
    #print('writing ix',ix)
    #with open('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprotID_'+str(ix)+'.txt','w') as f:
    #    for uid in subset['externalID'][starts[ix]:stops[ix]]:
    #        f.write(str(uid)+'\n')

#need to concatenate these chunks into one df
all_links = pd.DataFrame()
for ix in range(0,num_chunks):
    print(len(all_links))
    chunk = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprot2cds_'+str(ix)+'.tsv',sep='\t')
    all_links = all_links.append(chunk,ignore_index=True)
all_links.columns = ['externalID','EMBLcdsID']
all_links.to_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprot2cds.csv',index=False)

subsetcds = subset.merge(all_links,on='externalID') 
subsetcds.to_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes2cds_OGtaxa.csv',index=False)
'''

subsetcds = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes2cds_OGtaxa.csv')
OGtaxa = ['Sulfolobus','Streptomyces','Pseudomonas','Thermococcus','Bacillus','Methanosarcina','Acinetobacter','Mycobacterium','Lactobacillus','Enterococcus']

'''
#tried these to look at groups of OGs, ended up just looking for genera with lots of OGs so could be more fine-grained, see OGtaxa 
methano = joinedtax[joinedtax['scientific_name']=='Methanosarcina']
rhodo = joinedtax[joinedtax['scientific_name']=='Rhodobacter']
bact = joinedtax[joinedtax['scientific_name']=='Bacteroides']
myco = joinedtax[joinedtax['scientific_name']=='Mycobacterium']

thermo = joinedtax[joinedtax['scientific_name']=='Thermoplasmata'] #do 1200
actino = joinedtax[joinedtax['scientific_name']=='Actinobacteria'] #do 2200
alpha = joinedtax[joinedtax['scientific_name']=='Alphaproteobacteria']
bacter = joinedtax[joinedtax['scientific_name']=='Bacteroidia']
clost = joinedtax[joinedtax['scientific_name']=='Clostridia']
'''

def sample_OGs(df, num_ogs=1000, seqs_per_og=5, success_ogs=[]):
    #find first <num_ogs> OGs with >= <seqs_per_og> unique orthoDB genes
    counts = Counter(df['OGID']).most_common()
    ogs = [x[0] for x in counts if x[0] not in success_ogs] #remove ogs already sampled
    gids = {}
    total_sampled = 0
    for og in ogs:
        if total_sampled < num_ogs:
            to_cycle = df[df['OGID']==og]
            unique_genes = list(set(to_cycle['orthoDB_geneID']))
            if len(unique_genes) >= seqs_per_og:
                #sample one sequence from each unique orthoDB gene
                seqs = []
                genes = pd.Series(unique_genes).sample(seqs_per_og) #randomly choose some genes to sample from this OG
                for gene in genes:
                    to_sample = to_cycle[to_cycle['orthoDB_geneID']==gene]
                    seq_to_add = list(to_sample['EMBLcdsID'].sample(1, random_state=525))[0]
                    seqs.append(seq_to_add)
                gids[og] = seqs
                total_sampled += 1
    return gids

def get_and_write(gids, og_path, success_ogs):
    #download the 
    error_ogs = []
    for og in gids.keys():
        #it sometimes happens where all 5 sequences are not retrievable for whatever reason; in this case, just skip and report
        #(I'll try again with differently randomly sampled EMBL IDs in the below for loop)
        try:
            gidstr = ','.join(gids[og])
            outpath = og_path+str(og)+'.fasta'
            #eutils_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id='+gidstr+'&rettype=fasta&retmode=text'
            query_url = 'https://www.ebi.ac.uk/ena/browser/api/fasta/'+gidstr+'?download=true'
            r = requests.get(query_url)
            #sometimes returns results and just skips one that doesn't exist - I'm just going to ignore that...
            #len(r.text.split('>')[1:]) != 5
            #sometimes returns json with 'Internal Server Error' if sequence doesn't exist; in that case, record as error_og
            if r.text.startswith('>'):
                with open(outpath,'wb') as f:
                    f.write(r.content)
                success_ogs.append(og)
            else:
                error_ogs.append(og)
        except:
            error_ogs.append(og)
    return success_ogs,error_ogs

def check_taxa(taxa):
    taxasubset = Counter(subsetcds[subsetcds['scientific_name']==taxa]['OGID']).most_common()
    number = len(taxasubset)
    print('number of OGs in taxa',taxa,':',number)
    print(taxasubset[1000:1005])


#OGtaxa = ['Pseudomonas','Thermococcus','Bacillus','Methanosarcina','Acinetobacter','Mycobacterium','Lactobacillus','Enterococcus']
for taxa in OGtaxa:
    print('processing',taxa,'...')
    #get list files figure out how many OGs already retrieved
    og_path = '/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG_seqs/'+str(taxa)+'/'
    files = [x for x in Path(og_path).resolve().iterdir()]  
    success_ogs = [x.stem for x in files]
    taxasubset = subsetcds[subsetcds['scientific_name']==taxa]
    while len(success_ogs) < 1000:
        print('trying again, success so far with',len(success_ogs),'ogs')
        num_ogs = 1000-len(success_ogs)
        gids = sample_OGs(taxasubset,num_ogs=num_ogs,seqs_per_og=5,success_ogs=success_ogs)
        print('sampled OGs:',list(gids.keys())[0:3])
        print('getting and writing seqs...')
        #run get_and_write for seqs, downloading them to a file with the name <og>.fasta in the folder /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG_seqs/<taxa>
        success_ogs,error_ogs = get_and_write(gids,og_path=og_path,success_ogs=success_ogs)
