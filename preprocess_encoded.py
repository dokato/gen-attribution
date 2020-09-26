#!/usr/bin/env python
# coding: utf-8

# # Word embeddings for nucleotide tokens
import numpy as np
import pandas as pd
import sklearn, pickle
import sentencepiece as sp
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
import itertools

import word2vec_models as wv

import importlib
importlib.reload(wv) # reload wv if its already loaded

datadir = '/Users/matthiastreder/data/deeplearn/genetic-engineering-attribution-challenge/'
resultsdir = datadir + '/results/'


# ## Preprocess encoded data
train = pd.read_csv(datadir+'train_values.csv')
train.head()

# encode tokens as integers
vocab = 1000
sp_model = sp.SentencePieceProcessor(model_file=datadir + f'm{vocab}.model')
encoded = [sp_model.encode(seq) for seq in train.sequence]
lens = [len(enc) for enc in encoded]
print('Encoded as', sp_model.vocab_size(), 'integers')

del sp_model

# for efficiency we concatenate all the sequences into one array and pad the borders with -1's
pad = 2
encoded = [[-1]*pad + enc + [-1]*pad for enc in encoded]

# flatten and convert to numpy
encoded = np.array(list(itertools.chain(*encoded)), dtype=np.int16)

# relative frequencies
un = np.unique(encoded)
rel_freq = np.array([np.count_nonzero(encoded==u) for u in un])/len(encoded)

t = 0.0001
print(f'Number of tokens with freq > {t} (they get subsampled): {np.count_nonzero(rel_freq > t)}/{len(un)}')

discard_probabilities = wv.get_discard_probabilities_from_labels(encoded, t=t)
discard_probabilities[encoded==-1] = 1  # we never want the padded -1's to be selected as targets

# replace the -1's by mirroring the sequence, e.g.
# 2, 3, 5, -1, -1  ->  2, 3, 5, 3, 2
pos = 0
for le in lens:
    encoded[pos:pos+pad] = encoded[pos+pad+pad:pos+pad:-1] # fill up -1's at the left border
    encoded[pos+le+pad:pos+le+2*pad] = encoded[pos+le+pad-2:pos+le-2:-1] # fill up -1's at the right border
    pos += le + 2*pad

# distribution for sampling negative targets in Skipgram
negative_sampling_distribution = wv.get_negative_sampling_distribution(encoded)

with open(resultsdir + f'word2vec_train_data_x_vocab{vocab}.pickle', 'wb') as f:
    pickle.dump((encoded, discard_probabilities, t, negative_sampling_distribution), f)

