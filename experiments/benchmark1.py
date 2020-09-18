from pathlib import Path
from itertools import permutations
import sys
sys.path.insert(0, '..')
from utils import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

from sklearn.ensemble import RandomForestClassifier

DATA_DIR =  '/data/deeplearn/genetic-engineering-attribution-challenge/'
train_values = pd.read_csv(DATA_DIR + 'train_values.csv', index_col='sequence_id')
train_labels = pd.read_csv(DATA_DIR + 'train_labels.csv', index_col='sequence_id')
test_values = pd.read_csv(DATA_DIR + 'test_values.csv', index_col='sequence_id')

sequence_lengths = train_values.sequence.apply(len)

train_values.iloc[:, 1:].apply(pd.value_counts)

sorted_binary_features = train_values.iloc[:, 1:].mean().sort_values()

lab_ids = pd.DataFrame(train_labels.idxmax(axis=1), columns=['lab_id'])

bases = set(''.join(train_values.sequence.values))


n = 3
subsequences = [''.join(permutation) for permutation in permutations(bases, r=n)]

ngram_features = get_ngram_features(train_values, subsequences)

all_features = ngram_features.join(train_values.drop('sequence', axis=1))
train_no_sequence = train_values.drop('sequence', axis=1)

X = all_features #ngram_features

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

print('Score: ', rf.score(X, y))

print('Score (top10): ', top10_accuracy_scorer(rf, X, y))


