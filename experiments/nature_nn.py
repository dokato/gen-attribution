import time
import sys
sys.path.insert(0, '..')
from utils import *

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Input, Reshape
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD

m = np.load('split1.npz')
X_train, X_test, X_val = m['X_train'], m['X_test'], m['X_val']
y_train, y_test, y_val = m['y_train'], m['y_test'], m['y_val']
del m
dna_seqpad_length = 2*max_seq_length+cnn_filter_length

num_classes = len(train_labels.columns)

total_epoch = 100
filter_num = 128
filter_len = cnn_filter_length
num_dense_nodes = 200

model = Sequential()
model.add(Input((None, 4, dna_seqpad_length)))
model.add(Reshape(target_shape=(dna_seqpad_length, 4)))
model.add(Conv1D(input_shape=(dna_seqpad_length, 4), filters=filter_num, kernel_size=filter_len, activation="relu", padding ="same"))
model.add(MaxPooling1D(pool_size=8))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(num_dense_nodes, input_shape=(filter_num,)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
print(model.summary())

print('Start training... ')
# what is class weight?
history = model.fit(X_train, y_train, batch_size = 8, \
            validation_data=(X_val, y_val), nb_epoch=1, verbose=1)

print('Training finished')
model.save("first_model")