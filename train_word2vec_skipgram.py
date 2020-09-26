import os, sys, random
sys.path.append('/Users/matthiastreder/mt03/python_tools/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import sqrt, log

from word2vec_models import Word2Vec_SkipGram

import pandas as pd 
import numpy as np
import file_tools as ft   # MT

print("Torch", torch.__version__)

# CPU or GPU?
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

print('Using', dev.upper(), 'device')

device = torch.device(dev)  
torch.cuda.empty_cache()
    
datadir = '/data/deeplearn/genetic-engineering-attribution-challenge/'  #desktop
datadir = '/Users/matthiastreder/data/deeplearn/genetic-engineering-attribution-challenge/'  # laptop
resultsdir = datadir + 'results/'
modelsdir = datadir + 'models/'

# Hyperparameters
batch_size = 128
n_epochs = 10
lr = 0.01

window_size = 2     # size of local window left and right of target from where positive examples are picked
n_pos = 2*window_size # nr of positive targets
n_neg = 20    # nr of negative targets (5-20 works well for smaller datasets)

# embedding_size=50
embedding_size=100
embedding_size=300

# load train data
vocab = 1000 # vocabulary size of byte encoding algorithm
n_inputs = n_outputs = vocab    # vocabulary size

filename = f'word2vec_train_data_vocab{vocab}.pickle'
encoded, discard_probabilities, t, negative_sampling_distribution = ft.load_pickle(resultsdir + filename)

# convert numopy arrays to torch
encoded = torch.from_numpy(encoded.astype(np.int64)).to(device)
discard_probabilities = torch.from_numpy(discard_probabilities.astype(np.int64))
negative_sampling_distribution = torch.from_numpy(negative_sampling_distribution.astype(np.int64)).to(device)

# Instantiate network and optimizer
net = Word2Vec_SkipGram(n_inputs=n_inputs, n_outputs=n_outputs, embedding_size=embedding_size)
net.init_embeddings()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

n_encoded = encoded.size()[0]
n_neg_sampling = negative_sampling_distribution.numpy().shape[0]

u = torch.zeros((batch_size, 1), dtype=torch.long).to(device)  # scene input
v_pos = torch.zeros((batch_size, n_pos), dtype=torch.long).to(device) # target objects
v_neg = torch.zeros((batch_size, n_neg), dtype=torch.long).to(device) # non-target objects

losses = np.zeros((n_epochs,))

print(f'Starting training with vocab={vocab}, embedding size={embedding_size}')

# train one epoch = over all minibatches
for epoch in range(n_epochs):       # --- loop over epochs
    total_loss = 0
    
    # randomly discard some tokens for current iteration
    selected_token_ix = torch.nonzero(torch.rand(n_encoded) > discard_probabilities)
    selected_token_ix = selected_token_ix[torch.randperm(selected_token_ix.shape[0])] # suffle order
    # selected_tokens = encoded[selected_token_ix]
    # selected_tokens = selected_tokens[torch.randperm(selected_tokens.shape[0])] # suffle order

    # n_batches changes in every iteration
    n_batches = selected_token_ix.size()[0]//batch_size

    # shuffle negative examples
    negative_sampling_distribution = negative_sampling_distribution[torch.randperm(len(negative_sampling_distribution))]
    negative_pos = 0 # we cycle through the negative examples

    for batch in range(n_batches):  # --- loop over batches

        if batch % 1000 == 0: print('batch', batch)
        # populate train tensor with scenes (and map from images to actualy scene categories)
        sel = selected_token_ix[batch*batch_size:(batch+1)*batch_size]
        u[:] = encoded[sel]
        
        # sample POSITIVE examples (left/right neighbours)
        for col in range(window_size):
            v_pos[:, col] = encoded[sel-window_size+col][:, 0]
            v_pos[:, col+window_size] = encoded[sel+1+col][:, 0]

        # sample NEGATIVE examples
        if negative_pos+(batch_size+4)*n_neg > n_neg_sampling: negative_pos = 0
        for row in range(batch_size):
            negative_samples = negative_sampling_distribution[negative_pos:negative_pos+3*n_neg]
            # we have to exclude positive examples from the negative samples (if any)
            combined = torch.cat((u[row], v_pos[row, :], negative_samples))
            uniques, counts = combined.unique(return_counts=True)
            uniques = uniques[counts == 1]
            v_neg[row, :] = uniques[torch.randperm(uniques.size()[0])][:n_neg]
            negative_pos += n_neg

        optimizer.zero_grad()

        # outputs: [predictions, 1]
        loss = net(u, v_pos, v_neg)

        # calculate gradients and update
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()

    losses[epoch] = total_loss/n_batches
    print(f"epoch: {epoch}, loss: {losses[epoch]}")

ft.save_pickle(os.path.join(modelsdir, f'word2vec_skipgram_vocab{vocab}_{embedding_size}dim_{epoch+1}epochs.pickle'), net.state_dict())
ft.save_numpy(os.path.join(resultsdir, f'word2vec_skipgram_loss_vocab{vocab}_{embedding_size}dim_{epoch+1}epochs'), losses)

# save model
# torch.save(net.state_dict(), os.path.join(datadir,'models',f'word2vec_skipgram'))
print('Finished all.')
