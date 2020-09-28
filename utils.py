import numpy as np
import pandas as pd
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

def top10_accuracy_scorer_binary(estimator, X, y):
    """A modifiedcustom scorer that evaluates a model on whether the correct
    label is in the top 10 most probable predictions.

    Args:
        estimator (model, sklearn or keras): A model that should be evaluated.
        X (numpy array): The test data.
        y (numpy array): The ground truth matrix one hot encoded

    Returns:
        float: Accuracy of the model as defined by the proportion of predictions
               in which the correct label was in the top 10. Higher is better.
    """
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