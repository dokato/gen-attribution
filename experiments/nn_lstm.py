import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

print("Torch", torch.__version__)
torch.manual_seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_CLASSES  = 1314

class LSTM_network(nn.Module):

    def __init__(self, embedding_weights, hidden_dim = 128):
        super(LSTM_network, self).__init__()
        emb_vocab, emb_dim = embedding_weights.shape
        self.emb = nn.Embedding.from_pretrained(embedding_weights)
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

def load_embeddings(path):
    '''
    Reads pretrained embeddings from the *path*.
    '''
    pass

def train(X, y, epochs = 1, batch = 10, lr = 0.01):
    '''
    IN:
      X - matrix of shape (nr of sequences, sequence length, nr of embeddings)
      y - matrix of shape (nr of sequences, number of classes)
      epochs (int) - nr of epochs
      batch (int) - nr of examples shown in batches
      lr (float) - learning rate
    '''
    net = LSTM_network(EMB_DIM)
    n_samples = X.shape[0]
    batch_samples = np.arange(n_samples)

    optimizer = optim.Adam(rnn.parameters(), lr = lr)

    for e in range(epochs):
        total_loss = 0
        val_loss = 0
        net.train()
        np.random.shuffle(batch_samples)
        for batch in range(n_samples//batch_size):
            batch_ids = batch_samples[int(i*batch):int(i*batch + batch_size)]
            X_train = X[batch_ids, :, :] # how to handle different sequence size ???
            y_train = torch.from_numpy(y[batch_ids, :])

            y_pred = net.forward(X_train)  
            loss = F.cross_entropy(y_pred, y_train)

            # calculate gradients and update
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()

            total_loss += loss.item()
        net.eval()
        print(f"epoch: {e}, total loss: {total_loss}")
