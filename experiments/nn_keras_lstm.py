import os, sys, time, pickle
sys.path.insert(0, '..')
from utils import *
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, Input, Reshape, Embedding, LSTM
from tensorflow.keras.layers import MaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
import numpy as np

VOCAB_SIZE = 1000
N_CLASSES  = 1314

def make_model(max_seq_len, emb_size, lstm_hidden = 128):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE+1, emb_size, input_length = max_seq_len))
    model.add(Bidirectional(LSTM(lstm_hidden, return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_hidden)))
    model.add(Dense(1000))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    print(model.summary())

def train_model(model, X_train, y_train, X_val, y_val, epochs, save = False):
    cl_weight = get_class_weights(y_train)
    history = model.fit(X_train, y_train, batch_size = 8, \
                validation_data=(X_val, y_val), epochs = epochs, verbose=1,
                class_weight = cl_weight)
    if save:
        with open(save, 'wb') as f:
            pickle.dump(history, save + '.pkl')
        model.save(save + '.h5')

if __name__ == "__main__":
    NR_EPOCHS = 10
    BATCH_SIZE = 32
    MAX_LEN = 10000
    EMB_SIZE = 200
    print("Loading data")
    (X_train, y_train, X_test, y_test, X_val, y_val) = load_sequence_train_data(alpha = 1)
    sp = load_bpe_model(f'x{VOCAB_SIZE}.model')
    print('Encoding BPE')
    t0 = time.time()
    X_train = sp.encode(X_train)
    X_test = sp.encode(X_test)
    X_val = sp.encode(X_val)
    print('(took', (time.time() - t0)/60, ')')
    print('padding')
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=VOCAB_SIZE, padding='post', maxlen=MAX_LEN)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=VOCAB_SIZE, padding='post', maxlen=MAX_LEN)
    X_val = keras.preprocessing.sequence.pad_sequences(X_val, value=VOCAB_SIZE, padding='post', maxlen=MAX_LEN)
    print('padding done')
    print('training')
    net = make_model(MAX_LEN, EMB_SIZE, 128)
    train_model(net, X_train, y_train, X_val, y_val, NR_EPOCHS, 'klstm_1')