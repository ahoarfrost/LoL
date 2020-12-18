from Bio import SeqIO
from Bio import SeqRecord   
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--metagenome_id",type=str)
parser.add_argument("--max_seqs",type=int)
args = parser.parse_args()

metagenome_id = args.metagenome_id
max_seqs = args.max_seqs

fout = '/scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/'+str(metagenome_id)+'_cut20M.fastq'
ix=0
records = []
with open(fout, 'a') as outfile:
    for record in SeqIO.parse('/scratch/ah1114/LoL/TransferLearningTasks/EC1/TARA_metagenomes/'+str(metagenome_id)+'.fastq','fastq'):
        ix += 1
        if ix <= max_seqs:
            if ix % 1000000 == 0:
                print('processing ix',ix, 'out of',max_seqs)
            records.append(record)

    SeqIO.write(records, outfile, "fastq")


