from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
from pathlib import Path
    

def chunk_cds(fin, fout, genome_name, distrib, mode="w", shuffle=True):
    #fin should be a fasta file you want to process
    #fout is the name of the clean csv file you want to put into your model
    #distrib is an array of sequence lengths that reflect the distribution of seqlens you want in your output
    
    frames = {0:1, 1:3, 2:2}
    num_gene = 0
    num_extracted = 0
    gene_lens = []

    with open(fout, mode) as outfile:
        for fna in SeqIO.parse(fin, "fasta"):
            
            start = np.random.choice(range(0,10)) #pick first start position as somewhere near the beginning, but will only see full start codon 1/10 of the time
            records = []

            #decide how many subsequences to draw. I am naively doing this for now
            #n_sub = len(cds.seq)//135 #e.g. a 600bp gene will have 4 subsequences drawn from it at random
            num_gene += 1
            gene_lens.append(len(fna.seq))
            
            while start < len(fna.seq):
                seqlen = int(np.random.choice(distrib))
                end = start+seqlen
                sequence = fna.seq[start:end]
                if len(sequence)<60:
                    start = start + seqlen
                    continue
                else:
                    #IRL, gene is equally likely to be sequenced in either forward or reverse direction. Toss a coin which to use
                    use_complement = np.random.choice([0,1])
                    if use_complement:
                        subseq = sequence.reverse_complement()
                        #figure out frame and add that label to the fasta header
                        subframe = frames[start % 3] * -1
                        #create name for fasta header
                        subdescription = fna.name+"[subset="+str(start)+".."+str(end)+"] [frame="+str(subframe)+"]"
                    else:
                        subseq = sequence
                        subframe = frames[start % 3] 
                        subdescription = fna.name+"[subset="+str(start)+".."+str(end)+"] [frame="+str(subframe)+"]"

                    record = SeqRecord(subseq, id=genome_name,description=subdescription) 
                    records.append(record)
                    num_extracted += 1
                
                    start = end

            #write line to fasta fout
            if shuffle:
                #shuffle the records first
                np.random.shuffle(records)
            SeqIO.write(records, outfile, "fasta")

    print("--wrote",num_extracted,"records from",num_gene,"genes with an avg gene length of",np.mean(gene_lens),"to",fout)

root = Path('/scratch/ah1114/LoL/data/')
files = [x for x in (root/'GTDBrepCDS').resolve().iterdir()]
distrib = np.load('/scratch/ah1114/LoL/data/GTDBGenomeSequencingDistrib.npy')

for ix,fin in enumerate(files):
    genome_name = fin.name.split('_cds_from_genomic.fna')[0] 
    print('processing file', ix, 'out of', len(files),'for genome',genome_name,'...')
    fout = (root/'GTDBrepCDS_chunked'/str(genome_name+'.fna')).resolve()

    #if fout doesn't already exist, chunk_fna
    if fout.is_file():
        print('--file already exists, skipping')
    else:
        chunk_cds(fin=fin, fout=fout, genome_name=genome_name, distrib=distrib)
 