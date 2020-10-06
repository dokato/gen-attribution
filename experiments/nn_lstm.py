import os, sys, time
sys.path.insert(0, '..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


print("Torch", torch.__version__)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB_SIZE = 1000
N_CLASSES  = 1314

class LSTM_network(nn.Module):

    def __init__(self, embedding_weights, hidden_dim = 128):
        '''
        embedding_weights - torch.LongTensor (nr of emb, emb dim)
        '''
        super(LSTM_network, self).__init__()
        emb_vocab, emb_dim = embedding_weights.shape
        self.emb = nn.Embedding.from_pretrained(embedding_weights) # size of input 
        self.lstm = nn.LSTM(input_size = emb_dim,
                            hidden_size = int(hidden_dim),
                            num_layers = 2,
                            batch_first = True,
                            bidirectional = True)
        self.fc1 = nn.Linear(2*hidden_dim, 1000)
        self.drop = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1000, N_CLASSES)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        '''
        X - (batch size, seq. length)
        '''
        x_emb = self.emb(X) # output should be (batch size, seq. length, embedding size)
        x, _ = self.lstm(x_emb)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.softmax(self.fc2(x))
        return x

def padding_training_seq(X, vocab_size = VOCAB_SIZE):
    '''
    Padding training sequence to the size size.
    Returns torch.tensor.
    '''
    lengths = map(len, X)
    max_seq = max(lengths)
    n_samples = len(X)
    Xt = torch.zeros((n_samples, max_seq), device=device) + vocab_size
    for i in range(n_samples):
        Xt[i,:len(X[i])] = torch.tensor(X[i])
    return Xt.long()

def train(X, y, Xval, yval, emb_weights, Xtest = None, ytest = None,
          epochs = 1, batch = 512, lr = 0.01, save = None):
    '''
    IN:
      X - matrix of shape (nr of sequences, sequence length, nr of embeddings)
      y - matrix of shape (nr of sequences, number of classes)
      Xval, yval, Xtest = None, ytest = None - the same for validation and test
      emb_weights (torch.Tensor) - embedding weights
      epochs (int) - nr of epochs
      batch (int) - nr of examples shown in batches
      lr (float) - learning rate
      save (str) - if not None, then path to file where weights are saved
    '''
    net = LSTM_network(emb_weights, 200)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr = lr)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    Xval = padding_training_seq(Xval)
    yval = torch.from_numpy(yval).long()
    Xval, yval = Xval.to(device), yval.to(device)
    print('Training started')
    torch.cuda.empty_cache()
    for e in range(epochs):
        total_loss = 0
        val_loss = 0
        net.train()
        for xb, yb in batch_sorted(X, y, batch):
            X_train = padding_training_seq(xb)
            y_train = torch.from_numpy(yb).long()

            X_train, y_train = X_train.to(device), y_train.to(device)
            print(X_train.shape)
            print(torch.cuda.memory_allocated()/1024**2) #in Mb
            optimizer.zero_grad()

            y_pred = net.forward(X_train)
            loss = criterion(y_pred, y_train.argmax(1))

            # calculate gradients and update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('.', end = "")
            #torch.cuda.empty_cache()
        print('val')
        net.eval()
        y_pred_val = torch.zeros((Xval.shape[0], N_CLASSES), device=device)
        for i in range(Xval.shape[0]):
            a = net.forward(Xval[i, :].view((1, Xval.shape[1])))
            y_pred_val[i,:] = a
        loss = criterion(y_pred_val, yval.argmax(1))
        val_loss = loss.item()
        print(f"E: {e+1}/{epochs}| total training loss: {total_loss} | val loss: {val_loss}")
    print('Training done!')
    if Xtest:
        torch.cuda.empty_cache()
        ytest = torch.from_numpy(ytest).long()
        ytest = ytest.to(device)
        net.eval()
        y_pred_tst = torch.zeros((len(Xtest), N_CLASSES))
        for i in range(len(Xtest)):
            xt_ = padding_training_seq([Xtest[i]])
            print('>', xt_.shape)
            xt_ = xt_.to(device)
            a = net.forward(xt_.view((1, xt_.shape[1])))
            y_pred_tst[i,:] = a
        loss = criterion(y_pred_tst, ytest.argmax(1))
        tst_loss = loss.item()
        print(f'Test loss {tst_loss}')
        print('TOP 10 on test:', top10_accuracy_scorer_binary(y_pred_tst, None, ytest, proba=True))

    if save:
        torch.save(net.state_dict(), save)
        print('Save and exit')

if __name__ == "__main__":
    NR_EPOCHS = 2
    BATCH_SIZE = 8
    print("Loading data")
    (X_train, y_train, X_test, y_test, X_val, y_val) = load_sequence_train_data(alpha = 20)
    sp = load_bpe_model(f'x{VOCAB_SIZE}.model')
    print('Encoding BPE')
    t0 = time.time()
    X_train = sp.encode(X_train)
    X_test = sp.encode(X_test)
    X_val = sp.encode(X_val)
    print('(took', (time.time() - t0)/60, ')')
    print('Loading embeddings')
    embs = load_embeddings('word2vec_skipgram_vocab1000_100dim_10epochs.pickle')
    train(X_train, y_train, X_val, y_val, embs, X_test, y_test,
          epochs = NR_EPOCHS, batch=BATCH_SIZE)
