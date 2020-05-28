from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
from pathlib import Path
    

def chunk_fna(fin, fout, genome_name, distrib, mode="w", shuffle=True):
    #fin should be a fasta file you want to process
    #fout is the name of the clean csv file you want to put into your model
    #distrib is an array of sequence lengths that reflect the distribution of seqlens you want in your output
    
    with open(fout, mode) as outfile:
        for fna in SeqIO.parse(fin, "fasta"):
            
            start = 0
            records = []
            
            while start < len(fna.seq):
                seqlen = int(np.random.choice(distrib))
                #IRL, gene is equally likely to be sequenced in either forward or reverse direction. Toss a coin which to use
                use_complement = np.random.choice([0,1])
                if use_complement:
                    sequence = fna.seq[start:start+seqlen]
                    subseq = sequence.reverse_complement()
                    #create name for csv
                    subdescription = "[subset=reverse"+str(start)+".."+str(start+seqlen)+"]"
                else:
                    subseq = fna.seq[start:start+seqlen]
                    subdescription = "[subset="+str(start)+".."+str(start+seqlen)+"]"

                record = SeqRecord(subseq, id=genome_name,description=subdescription) 
                records.append(record)
                
                start = start + seqlen

            #write line to fasta fout
            if shuffle:
                #shuffle the records first
                np.random.shuffle(records)
            SeqIO.write(records, outfile, "fasta")

root = Path('/scratch/ah1114/LoL/data/')
files = [x for x in (root/'PDB_to_ENA_seqs').resolve().iterdir()]
distrib = np.load('/scratch/ah1114/LoL/data/GTDBGenomeSequencingDistrib.npy')

for ix,fin in enumerate(files):
    print('processing file', ix, 'out of', len(files),'...')
    genome_name = fin.name.split('.fasta')[0] 
    fout = (root/'PDB_to_ENA_chunked'/str(genome_name+'.fasta')).resolve()

    #if fout doesn't already exist, chunk_fna
    if fout.is_file():
        print('--file already exists, skipping')
    else:
        chunk_fna(fin=fin, fout=fout, genome_name=genome_name, distrib=distrib)
