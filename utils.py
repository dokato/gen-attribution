import numpy as np
import pandas as pd
import itertools
import time
import os
import pickle
import torch
import scipy.stats as st

import sentencepiece as spm

## contants

DATA_DIR =  "/data/deeplearn/genetic-engineering-attribution-challenge/"
EMB_PATH = "/data/deeplearn/genetic-engineering-attribution-challenge/models/"
BPE_PATH = "/home/guest/dominik/gen-attribution/weights"

## auxiliary function

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
    full_seqs = [''] * len(seqs)
    rc_dict = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
    for j in range(len(seqs)):
        fwd_seq = seqs[j]
        complement_seq = ''
        for n in fwd_seq:
            complement_seq += rc_dict[n]
        full_seqs[j] = fwd_seq + 'N'*filter_length + complement_seq[::-1] #[::-1] reverses string 
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


def get_ngram_features(data, subsequences):
    """Generates counts for each subsequence.

    Args:
        data (DataFrame): The data you want to create features from. Must include a "sequence" column.
        subsequences (list): A list of subsequences to count.

    Returns:
        DataFrame: A DataFrame with one column for each subsequence.
    """
    features = pd.DataFrame(index=data.index)
    for subseq in subsequences:
        features[subseq] = data.sequence.str.count(subseq)
    return features

def top10_accuracy_scorer(estimator, X, y):
    """A custom scorer that evaluates a model on whether the correct label is in 
    the top 10 most probable predictions.

    Args:
        estimator (sklearn estimator): The sklearn model that should be evaluated.
        X (numpy array): The validation data.
        y (numpy array): The ground truth labels.

    Returns:
        float: Accuracy of the model as defined by the proportion of predictions
               in which the correct label was in the top 10. Higher is better.
    """
    # predict the probabilities across all possible labels for rows in our training set
    probas = estimator.predict_proba(X)
    
    # get the indices for top 10 predictions for each row; these are the last ten in each row
    # Note: We use argpartition, which is O(n), vs argsort, which uses the quicksort algorithm 
    # by default and is O(n^2) in the worst case. We can do this because we only need the top ten
    # partitioned, not in sorted order.
    # Documentation: https://numpy.org/doc/1.18/reference/generated/numpy.argpartition.html
    top10_idx = np.argpartition(probas, -10, axis=1)[:, -10:]
    
    # index into the classes list using the top ten indices to get the class names
    top10_preds = estimator.classes_[top10_idx]

    # check if y-true is in top 10 for each set of predictions
    mask = top10_preds == y.reshape((y.size, 1))
    
    # take the mean
    top_10_accuracy = mask.any(axis=1).mean()
 
    return top_10_accuracy

def top10_accuracy_scorer_binary(estimator, X, y, proba = False):
    """A modifiedcustom scorer that evaluates a model on whether the correct
    label is in the top 10 most probable predictions.

    Args:
        estimator (model, sklearn or keras, matrix): A model that should be
              evaluated, or matrix with probabilities (look *proba*)
        X (numpy array): The test data.
        y (numpy array): The ground truth matrix one hot encoded
        proba (bool) - if True then estimator is assumed to be probabilities
    Returns:
        float: Accuracy of the model as defined by the proportion of predictions
               in which the correct label was in the top 10. Higher is better.
    """
    if proba:
        probas = estimator
    else:
        probas = estimator.predict_proba(X)
    top10_idx = np.argpartition(probas, -10, axis=1)[:, -10:]
    y_real_idx = np.where(y==1)[1]
    mask = top10_idx == y_real_idx.reshape((y_real_idx.size, 1))
    return mask.any(axis=1).mean()

def get_class_weights(y_train):
    """Generates class weights.

    Args:
        y_train (np.array): one-hot encoded labels

    Returns:
        dict: with class weights
    """
    cl_weight = {}
    for i in range(y_train.shape[1]):
        cl_weight[i] = 0
    for x in range(y_train.shape[0]):
        cl_weight[np.argmax(y_train[x,:])] += 1
    sumval = sum(cl_weight.values())
    for y in cl_weight.keys():
        cl_weight[y] = len(cl_weight)*float(cl_weight[y])/float(sumval)
    return cl_weight

def batch_sorted(X, y, batch_size):
    '''
    It returns sorted values of X by length of sequence:
    IN:
      X (list) -  training data (nr examples, seq length)
      y (np.array/list) - label matrix
      batch_size (int) - nr of examples in batch
    OUT:
      iterator, giving (X_batch, y_batch) tuple
    '''
    assert len(X) == len(y), "X and y not the same size"
    X = list(X)
    n_tr = len(X)
    lengths = map(len, X)
    idcs_sorted = [ii[1] for ii in sorted(zip(lengths, range(n_tr)))]
    buckets = []
    bucket_idcs = []
    X = np.array(X, dtype=object)
    y = np.asarray(y)
    while len(idcs_sorted):
        bucket_idcs.append(idcs_sorted.pop(0))
        if len(bucket_idcs) == batch_size:
            np.random.shuffle(bucket_idcs)
            buckets.append((X[bucket_idcs], y[bucket_idcs]))
            bucket_idcs = []
    np.random.shuffle(bucket_idcs)
    buckets.append((X[bucket_idcs], y[bucket_idcs]))
    np.random.shuffle(buckets)
    for Xb, yb in buckets:
        yield Xb, yb

def load_embeddings(emb_file, path = EMB_PATH, add_padding = 'zero'):
    '''
    Reads pretrained embeddings from the *path*/*emb_file*.
    If *add_padding* is not None, it adds extra embedding (last index) for padding:
    'zero' - full of zeros
    'mean' - average of all embeddings
    '''
    with open(os.path.join(path, emb_file), 'rb') as f: 
        embw = pickle.load(f)
        embw = embw[0]
    embeddings = 0.5*(embw['input_embeddings.weight'] + embw['output_embeddings.weight'])
    nr_emb, emb_dim = embeddings.shape
    if add_padding == 'zero':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        padz = torch.zeros((1, emb_dim), device = device)
        embeddings = torch.cat((embeddings, padz), 0)
    if add_padding == 'mean':
       embeddings = torch.cat((embeddings, embeddings.mean(axis=0).view((1, emb_dim))), 0)
    return embeddings

def load_bpe_model(model_file, path = BPE_PATH):
    sp = spm.SentencePieceProcessor(model_file = os.path.join(path, model_file))
    return sp

def load_sequence_train_data(train_split = 0.8, test_split = 0.15, val_split = 0.05, alpha = 5):
    '''
    Loads sequence data from genetic attribution challange.
    IN:
      train_split (float) - portion fo data for training
      test_split (float) - portion fo data for testing
      val_split (float) - portion fo data for validation
      alpha (int) - percent of values to filter based on seq length
    OUT:
      (X_train, y_train, X_test, y_test, X_val, y_val) - X_* list of sequences, y_* matrix with one hot encoded label
    '''
    assert train_split + test_split + val_split <= 1, "split proportion must add up to 1"
    train_values = pd.read_csv(DATA_DIR + 'train_values.csv', index_col='sequence_id')
    train_labels = pd.read_csv(DATA_DIR + 'train_labels.csv', index_col='sequence_id')
    train_labels_onehot = train_labels.to_numpy()
    seqs = list(train_values.sequence)
    seqs = np.array(seqs, dtype=object)
    # due to GPU memory issues one needs to filter out very long sequences
    lens = np.array(list(map(len, seqs)))
    seqs = seqs[lens < st.scoreatpercentile(lens, 100-alpha)]
    n_seq = len(seqs)
    idxs = np.arange(n_seq)
    np.random.shuffle(idxs)
    tr_idxs = idxs[:int(train_split*n_seq)]
    tst_idxs = idxs[int(train_split*n_seq):int(train_split*n_seq)+int(test_split*n_seq)]
    val_idxs = idxs[int(train_split*n_seq)+int(test_split*n_seq):]
    X_train = seqs[tr_idxs]
    X_test = seqs[tst_idxs]
    X_val = seqs[val_idxs]
    y_train = train_labels_onehot[tr_idxs,:]
    y_test = train_labels_onehot[tst_idxs,:]
    y_val = train_labels_onehot[val_idxs,:]
    return (list(X_train), y_train, list(X_test), y_test, list(X_val), y_val)

def trim_sequences(X, y, trim):
    '''
    Makes sure that sentences in X are mamximum *trim* size.
    It cuts them into pieces and adds trailing to the end of the list.
    '''
    Xn = []
    yn = []
    y = list(y)
    for e, sq in enumerate(X):
        if len(sq) < trim:
            Xn.append(sq)
            yn.append(y[e])
        else:
            for i in range(0, len(seq), trim):
                Xn.append(sq[i*trim:((i+1)*trim)])
                yn.append(y[e])
    return Xn, np.array(yn)

def load_trimmed_sequence_train_data(train_split = 0.8, test_split = 0.15, val_split = 0.05, trim = 2000):
    '''
    Loads sequence data from genetic attribution challange.
    IN:
      train_split (float) - portion fo data for training
      test_split (float) - portion fo data for testing
      val_split (float) - portion fo data for validation
      trim (int) - percent of values to filter based on seq length
    OUT:
      (X_train, y_train, X_test, y_test, X_val, y_val) - X_* list of sequences, y_* matrix with one hot encoded label
    '''
    assert train_split + test_split + val_split <= 1, "split proportion must add up to 1"
    train_values = pd.read_csv(DATA_DIR + 'train_values.csv', index_col='sequence_id')
    train_labels = pd.read_csv(DATA_DIR + 'train_labels.csv', index_col='sequence_id')
    train_labels_onehot = train_labels.to_numpy()
    seqs = list(train_values.sequence)
    n_seq = len(seqs)
    idxs = np.arange(n_seq)
    np.random.shuffle(idxs)
    tr_idxs = idxs[:int(train_split*n_seq)]
    tst_idxs = idxs[int(train_split*n_seq):int(train_split*n_seq)+int(test_split*n_seq)]
    val_idxs = idxs[int(train_split*n_seq)+int(test_split*n_seq):]
    X_train = seqs[tr_idxs]
    X_test = seqs[tst_idxs]
    X_val = seqs[val_idxs]
    y_train = train_labels_onehot[tr_idxs,:]
    y_test = train_labels_onehot[tst_idxs,:]
    y_val = train_labels_onehot[val_idxs,:]
    X_train, X_test, X_val = list(X_train), list(X_test), list(X_val)
    X_train_n = trim_sequences(X_train, trim)
    X_test_n = trim_sequences(X_test, trim)
    X_val_n = trim_sequences(X_val, trim)
    return (X_train_n, y_train, X_test_n, y_test, X_val_n, y_val)
