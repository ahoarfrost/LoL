import pandas as pd
import requests
from collections import Counter
import urllib.parse 
import urllib.request 
from pathlib import Path

#to choose OGs at the class level, get cds ids and convert to subsetcds:
#look for classes with at least a few tens of thousands of rows (at least 1000 unique OGIDs)
joinedtax = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes_withtaxa.csv',index_col=False)

#Counter(joinedtax['scientific_name']).most_common()   #can choose manually from this
#I chose the classes for the genera used in earlier iteration: OGtaxa = ['Sulfolobus','Streptomyces','Pseudomonas','Thermococcus','Bacillus','Methanosarcina','Acinetobacter','Mycobacterium','Lactobacillus','Enterococcus']
#Acinetobacter are also in gammaproteobacteria, and mycobacterium also in Actinobacteria, lactobacillus also in Bacilli, enterococcus also in Bacilli; Thermococci wasn't used in OG groups;, so chose additional classes

OGclassnames = ['Thermoprotei','Actinobacteria','Gammaproteobacteria','Spirochaetia','Bacilli','Methanomicrobia','Halobacteria','Bacteroidia','Deltaproteobacteria','Clostridia'] 
#actinobacteria is both a class and a phylum so changed this to NCBI_taxIDs
OGclass = [183924,1760,1236,203692,91061,224756,183963,200643,28221,186801]

subset = joinedtax[joinedtax['NCBI_taxID'].isin(OGclass)] 

print('linking to EMBL CDS ids...')
#map the uniprot IDs we care about to EMBL/GenBank/DDBJ CDS sequences - the uniprot database mapper is good for this
num_chunks = 100000 #have to do in chunks or api crashes
interval = int(len(subset)/num_chunks)+1
starts = [x*interval for x in range(0,num_chunks)]
stops = [x*interval for x in range(1,num_chunks+1)] 
url = 'https://www.uniprot.org/uploadlists/'
fout = '/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprot2cds_class/uniprot2cds_class.tsv'
for ix in range(0,num_chunks):
    #query uniprot database mapper API, see: https://www.uniprot.org/help/api_idmapping
    #if fout doesn't exist, continue
    #if not Path(fout).is_file():
    if ix % 1000 == 0:
        print('querying ix',ix,'out of',num_chunks)

    query = ' '.join(subset['externalID'][starts[ix]:stops[ix]])
    params = { 
    'from': 'ACC+ID', #uniprot KB id
    'to': 'EMBL', #EMBL/GenBank/DDBJ CDS
    'format': 'tab', 
    'query': query
    } 
    try:
        data = urllib.parse.urlencode(params) 
        data = data.encode('utf-8') 
        req = urllib.request.Request(url, data) 
        with urllib.request.urlopen(req) as f: 
            response = f.read() 
    except:
        print('error on ix',ix,'with query: ', data)
    #this will write tsv file with columns "From" and "To" - from is uniprot; to is embl cds
    if ix==0:
        with open(fout,'w') as f:
            f.writelines(response.decode('utf-8')) 
    else:
        with open(fout,'a') as f:
            f.writelines(response.decode('utf-8').split('From\tTo\n')[1])


#need to concatenate these chunks into one df
print('renaming colnames...')
rename = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprot2cds_class/uniprot2cds_class.tsv')
rename.columns = ['externalID','EMBLcdsID']
rename.to_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/uniprot2cds_class/uniprot2cds_class.tsv',index=False)

subsetcds = subset.merge(rename,on='externalID') 
subsetcds.to_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes2cds_OGtaxa_class.csv',index=False)

subsetcds = pd.read_csv('/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG2genes2cds_OGtaxa_class.csv')

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


print('getting and writing fasta...')
for taxa in OGclassnames:
    print('processing',taxa,'...')
    #get list files figure out how many OGs already retrieved
    og_path = '/scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG_seqs_class/'+str(taxa)+'/'
    files = [x for x in Path(og_path).resolve().iterdir()]  
    success_ogs = [x.stem for x in files]
    taxasubset = subsetcds[subsetcds['scientific_name']==taxa]
    while len(success_ogs) < 1000:
        print('trying again, success so far with',len(success_ogs),'ogs')
        num_ogs = 1000-len(success_ogs)
        gids = sample_OGs(taxasubset,num_ogs=num_ogs,seqs_per_og=5,success_ogs=success_ogs)
        print('sampled OGs:',list(gids.keys())[0:3])
        print('getting and writing seqs...')
        #run get_and_write for seqs, downloading them to a file with the name <og>.fasta in the folder /scratch/ah1114/LoL/TransferLearningTasks/Homologs_Emb/OG_seqs_class/<taxa>
        success_ogs,error_ogs = get_and_write(gids,og_path=og_path,success_ogs=success_ogs)
