import time
import sys
sys.path.insert(0, '..')
from utils import *

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.core import Dense, Activation, Flatten
from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers.convolutional import Convolution1D
from tensorflow.keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import SGD

X_train, y_train = 
X_val, y_val = 
X_test, y_test = 

dna_seqpad_length = X_train

num_classes = 

total_epoch = 
filter_num = 
filter_len = 
num_dense_nodes = 

model = Sequential()
model.add(Convolution1D(input_dim=4, input_length=dna_seqpad_length, nb_filter=filter_num, filter_length=filter_len, activation="relu", border_mode ="same"))
model.add(MaxPooling1D(pool_length=dna_seqpad_length))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(input_dim=filter_num,output_dim=num_dense_nodes))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(output_dim=num_classes))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
print(model.summary())

# what is class weight?
history = model.fit(X_train, y_train, batch_size = 8, \
            validation_data=(X_val, y_val), nb_epoch=1, verbose=1)

