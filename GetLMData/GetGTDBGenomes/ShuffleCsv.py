import pandas as pd
from pathlib import Path
import os
import datetime as dt

root = Path('/scratch/ah1114/LoL/data/GTDBrepGenomes_chunked_csv/')
files = [x for x in root.iterdir()]

#read in each file, shuffle the rows of the csv, and write to same filename
for ix,csv in enumerate(files):
    filetime = dt.datetime.fromtimestamp(os.path.getctime(csv)) 
    if filetime.date() >= dt.datetime(2020,1,21).date():
        print('skipping file',csv.name,' - already processed')
    else:
        print('processing file',ix,'out of',len(files),":",csv.name)
        old = pd.read_csv(csv)
        new = old.sample(frac=1)
        new.to_csv(csv, index=False)