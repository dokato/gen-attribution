import time
import sys
sys.path.insert(0, '..')
from utils import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
np.savez('encoding1.npz', enc_seqs=enc_seqs, train_labels_onehot=train_labels_onehot)

# this causes memory error
#X_train, X_test, y_train, y_test = train_test_split(enc_seqs, train_labels_onehot, test_size=0.10, random_state=12)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.70, random_state=12)
n_seq = enc_seqs.shape[0]
idxs = np.arange(n_seq)
np.random.shuffle(idxs)
train_split = 0.8
test_split = 0.1
val_split = 0.1

assert train_split + test_split + val_split <= 1, "split proportion must add up to 1"

tr_idxs = idxs[:int(train_split*n_seq)]
tst_idxs = idxs[int(train_split*n_seq):int(train_split*n_seq)+int(test_split*n_seq)]
val_idxs = idxs[int(train_split*n_seq)+int(test_split*n_seq):]
X_train = enc_seqs[tr_idxs,:,:]
X_test = enc_seqs[tst_idxs,:,:]
X_val = enc_seqs[val_idxs,:,:]
y_train = train_labels_onehot[tr_idxs,:]
y_test = train_labels_onehot[tst_idxs,:]
y_val = train_labels_onehot[val_idxs,:]

print('Split into test/training finished', (time.time()-t0)/60 ,'min')
print('Saving data...')
np.savez('split1.npz', X_train=X_train, X_test=X_test, X_val=X_val, \
                       y_train=y_train, y_test=y_test, y_val=y_val)
print('Finished')