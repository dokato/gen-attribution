from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

DATA_DIR =  '/data/deeplearn/genetic-engineering-attribution-challenge/'
train_values = pd.read_csv(DATA_DIR + 'train_values.csv', index_col='sequence_id')
train_labels = pd.read_csv(DATA_DIR + 'train_labels.csv', index_col='sequence_id')
test_values = pd.read_csv(DATA_DIR + 'test_values.csv', index_col='sequence_id')

sequence_lengths = train_values.sequence.apply(len)

train_values.iloc[:, 1:].apply(pd.value_counts)

sorted_binary_features = train_values.iloc[:, 1:].mean().sort_values()

lab_ids = pd.DataFrame(train_labels.idxmax(axis=1), columns=['lab_id'])

bases = set(''.join(train_values.sequence.values))

from itertools import permutations

n = 4
subsequences = [''.join(permutation) for permutation in permutations(bases, r=n)]

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

ngram_features = get_ngram_features(train_values, subsequences)

all_features = ngram_features.join(train_values.drop('sequence', axis=1))

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

from sklearn.ensemble import RandomForestClassifier

X = all_features

# Create our labels
y = lab_ids.values.ravel()

rf = RandomForestClassifier(
    n_jobs=4, 
    n_estimators=150,
    class_weight='balanced', # balance classes
    max_depth=3, # shallow tree depth to prevent overfitting
    random_state=0 # set a seed for reproducibility
)

# fit our model
rf.fit(X, y)

rf.score(X, y)

print(top10_accuracy_scorer(rf, X, y))


