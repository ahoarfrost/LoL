import pandas as pd
import urllib.request
import os

datasets = pd.read_csv("/scratch/ah1114/LoL/data/GenomeIDs_parsedclass_withftp.csv")

for ix, genome in datasets.iterrows():
    accession = genome['ncbi_genbank_assembly_accession']
    name = genome['ncbi_assembly_name']
    ftp = genome['ftp_path']

    url = str(ftp)+'/'+str(accession)+'_'+str(name)+'_genomic.fna.gz'
    outpath = '/scratch/ah1114/LoL/data/GTDBrepGenomes/'+str(accession)+'_'+str(name)+'_genomic.fna.gz'
    if os.path.isfile(outpath)==False:
        try:
            urllib.request.urlretrieve(url, outpath)
        except:
            print(ix,',',url)

print('Done!') 
