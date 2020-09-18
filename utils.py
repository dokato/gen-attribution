import numpy as np
import itertools
import time
import pickle

def pad_dna(seqs, maxlen):
    '''
    Pad DNA sequences with 'N' added at the end.
    IN:
     *seqs* - list
       list of character with DNA sequences.
     *maxlen* - int
       maximum length of the sequence
    OUT:
       list of padded DNA sequences
    '''
    start = time.time()
    padded_seqs = [''] * len(seqs)
    for i in seqs:
        if len(i) > maxlen:
            i = i[:maxlen]
            maxlen = len(i)
    for j in range(len(seqs)):
        if len(seqs[j]) > maxlen:
            seq = seqs[j][0:maxlen]
        else:
            seq = seqs[j]
        padded_seqs[j] = seq + "N" * (maxlen - len(seq))
    end = time.time()
    return padded_seqs

def append_rc(seqs, filter_length):
    '''
    Appends reversed sequence.
    IN:
     *seqs* - list
       list of character with DNA sequences.
     *filter_length* - int
       a gap between forward and backward sequence
    OUT:
       list of reversed dna seqs
    '''
    start = time.time()
    full_seqs = [''] * len(seqs)
    rc_dict = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
    for j in range(len(seqs)):
        fwd_seq = seqs[j]
        complement_seq = ''
        for n in fwd_seq:
            complement_seq += rc_dict[n]
        full_seqs[j] = fwd_seq + 'N'*filter_length + complement_seq[::-1] #[::-1] reverses string 
    end = time.time()
    return full_seqs

def convert_onehot_4(list_of_seqs):
    '''
    Converts DNA string sequence with ATGCN characters to 4 dim one-hot encoding.
    IN:
     *list_of_seqs* - list
       list of character with DNA sequences.
    OUT:
       list with one hot encoded sequences
    '''
    list_of_onehot2D_seqs = np.zeros((len(list_of_seqs),4,len(list_of_seqs[0])))
    nt_dict = {'A':[1,0,0,0], 'T':[0,1,0,0], 'G':[0,0,1,0], 'C':[0,0,0,1], 'N':[0,0,0,0]}
    count = 0
    for seq in list_of_seqs:
        if len(seq) > 1:
            for letter in range(len(seq)):
                for i in range(4):
                    list_of_onehot2D_seqs[count][i][letter] = (nt_dict[seq[letter]])[i]
        count += 1
    return list_of_onehot2D_seqs