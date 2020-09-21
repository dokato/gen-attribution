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

def get_discard_probabilities(y, t = 0.01):
    """Create a Word2vec-style subsampling distribution.
        
    Parameters:
        y (array): array of class labels
        t (float): threshold (default 0.01). 
    
    Creates a vector of probabilities that specifies how likely each sample 
    is discarded in each iteration. Classes with a frequency of <t are never discarded.

    Reference:
    Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). 
    Distributed Representations of Words and Phrases and their Compositionality. 
    In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. Q. Weinberger (Eds.), 
    Advances in Neural Information Processing Systems 26 (pp. 3111–3119). 
    Curran Associates, Inc. 
    Retrieved from http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    """
    unique = np.unique(y)
    counts = np.array([np.count_nonzero(y == u) for u in unique])
    class_frequency = counts / counts.sum()

    # Formula on p.4, Eq. 5: discard probability per class
    discard_class = 1 - np.sqrt(t/class_frequency)
    
    # now turn this into discard probability per sample
    discard_probabilities = np.zeros(len(y), dtype=np.float)
    for u, p in zip(unique, discard_class):
        discard_probabilities[y == u] = p
    
    return discard_probabilities

def discard_samples(discard_probabilities, y, X = None):
    """Randomly discard samples.
        
    Parameters:
        discard_probabilities (array): array of probabilities of discarding each sample
        y, X (arrays): class labels and features
    
    """
    select_ix = np.squeeze(np.array([np.random.rand(len(y)) > discard_probabilities]))
    print(select_ix)
    ys = y[select_ix]
    if X is not None:
        Xs = X[select_ix, :]
        return ys, Xs
    else:
        return ys
