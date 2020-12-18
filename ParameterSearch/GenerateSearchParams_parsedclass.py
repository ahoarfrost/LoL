import numpy as np
import pandas as pd
from pathlib import Path

print('ksize','stride','n_layers','n_hid','drop_mult','wd','moms','emb_sz','bptt','lrate','bs')
filename = Path('SearchParams.tsv')
numrounds=1000

for round in range(0,numrounds):
    ksize=1
    stride=1
    n_layers = int(np.random.choice([2,3,4]))
    n_hid = int(np.random.choice([576,1152]))
    drop_mult = float(np.random.choice([0.1,1.0]))
    wd = float(np.random.choice([1e-1,1e-2,1e-3]))

    mom_options = [(0.98,0.9),(0.9,0.8),(0.8,0.7)]
    moms = mom_options[np.random.choice(len(mom_options))]

    emb_sz = int(np.random.choice([56,104,128]))
    bptt = int(np.random.choice([75,100,125]))
    lrate = float(np.random.choice([2e-3,5e-3,8e-3]))
    bs = int(np.random.choice([512,1024]))

    print(ksize,stride,n_layers,n_hid,drop_mult,wd,moms,emb_sz,bptt,lrate,bs)

    params = [ksize,stride,n_layers,n_hid,drop_mult,wd,moms,emb_sz,bptt,lrate,bs]
    params = [str(param) for param in params]
    str_params = '\t'.join(params)
    filename.open('a').write(str_params+'\n')
