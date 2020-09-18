import time
import sys
sys.path.insert(0, '..')
from utils import *

import numpy as np
import pandas as pd

DATA_DIR =  '/data/deeplearn/genetic-engineering-attribution-challenge/'
train_values = pd.read_csv(DATA_DIR + 'train_values.csv', index_col='sequence_id')
train_labels = pd.read_csv(DATA_DIR + 'train_labels.csv', index_col='sequence_id')

max_seq_length = 8000
cnn_filter_length = 12

print('Preprocessing start...')
t0 = time.time()
train_labels_onehot = train_labels.to_numpy()
seqs = list(train_values.sequence)
enc_seqs = pad_dna(seqs, max_seq_length)
enc_seqs = append_rc(enc_seqs, cnn_filter_length)
enc_seqs = convert_onehot_4(enc_seqs)
print('One hot encoding finished')

X_train, X_test, y_train, y_test = train_test_split(enc_seqs, train_labels_onehot, test_size=0.10, random_state=12)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.70, random_state=12)
print('Split into test/training finished', (time.time()-t0)/60 ,'min')

np.savez('split1.npz', X_train=X_train, X_test=X_test, X_val=X_val, \
                       y_train=y_train, y_test=y_test, y_val=y_val)