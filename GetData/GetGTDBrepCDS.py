import pandas as pd
import urllib.request
import os

datasets = pd.read_csv("/scratch/ah1114/LoL/data/GTDBrepGenomes_withftp.csv")

for ix, genome in datasets.iterrows():
    accession = genome['# assembly_accession']
    name = genome['asm_name']
    ftp = genome['ftp_path']

    url = str(ftp)+'/'+str(accession)+'_'+str(name)+'_cds_from_genomic.fna.gz'
    outpath = '/scratch/ah1114/LoL/data/GTDBrepCDS/'+str(accession)+'_'+str(name)+'_cds_from_genomic.fna.gz'
    if os.path.isfile(outpath)==False: #if haven't downloaded the file yet, do it
        try:
            urllib.request.urlretrieve(url, outpath)
        except:
            #there aren't cds files for every genome; also might be other issues
            print(ix,',',url)

print('Done!') 
