from Bio import SeqIO
import numpy as np
import os


def count_seqs(directory):
    counts = []
    for filename in os.listdir(directory):
        count = 0
        for record in SeqIO.parse(directory+filename,"fasta"): 
            count += 1
        counts.append(count)
    return counts, np.sum(counts)

train_counts,total_train = count_seqs('/scratch/ah1114/LoL/data/GTDB_chunked_train/')
print('number of seqs in train folder:', total_train)
print('min, average, median, max of seqs in train folder:',np.min(train_counts),np.mean(train_counts),np.median(train_counts),np.max(train_counts))
valid_counts,total_valid = count_seqs('/scratch/ah1114/LoL/data/GTDB_chunked_valid/')
print('number of seqs in valid folder:', total_valid)
print('min, average, median, max of seqs in valid folder:',np.min(valid_counts),np.mean(valid_counts),np.median(valid_counts),np.max(valid_counts))
test_counts,total_test = count_seqs('/scratch/ah1114/LoL/data/GTDB_chunked_test/')
print('number of seqs in test folder:', total_test)
print('min, average, median, max of seqs in test folder:',np.min(test_counts),np.mean(test_counts),np.median(test_counts),np.max(test_counts))
