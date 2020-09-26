'''
Word2vec implementations in PyTorch and corresponding helper functions.

Classes:

    Word2Vec_CBOW
    Word2Vec_SkipGram

Functions:

    get_discard_probabilities(y, t)
    discard_samples(discard_probabilities, y, X)

'''
# @matthiastreder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import sqrt, log
import numpy as np


def get_discard_probabilities_from_labels(y, t = 0.01):
    """Create a Word2vec-style subsampling distribution from a vector of labels.
        
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
    discard_probabilities = np.zeros(len(y), dtype=np.float16)
    for u, p in zip(unique, discard_class):
        discard_probabilities[y == u] = p
    
    return discard_probabilities

def get_discard_probabilities_from_counts(count, t = 0.01):
    """Create a Word2vec-style subsampling distribution from a dictionary of counts
        
    Parameters:
        count (dict): dictionary of token-count pairs
        t (float): threshold (default 0.01). 
    
    Creates a vector of probabilities that specifies how likely each sample 
    is discarded in each iteration. Classes with a frequency of <t are never discarded.
  """
    raise NotImplementedError()

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

def get_negative_sampling_distribution(y):
    '''Negative sampling is used in the Skipgram model
    so that not all negative samples in the softmax layer 
    have to be activated and backpropagated, increasing efficiency.

    Parameters:
        y (array): array of class labels
    
    Returns:
        negative_sampling_distribution (array): array of labels which can be sampled
    '''
    # counts = {u : np.power(np.count_nonzero(y == u), 0.75) for u in np.unique(y)}
    labels = np.unique(y)
    negative_sampling_probabilities = np.array([np.count_nonzero(y == l) for l in labels], dtype=np.float)
    negative_sampling_probabilities = np.power(negative_sampling_probabilities, 0.75)
    negative_sampling_probabilities /= negative_sampling_probabilities.sum()
    negative_sampling_distribution = []
    for ix, l in enumerate(labels):
        n = np.ceil(negative_sampling_probabilities[ix]*10000.).astype(np.int)  # number of copies of current element
        negative_sampling_distribution.extend([l] * n)  

    return np.array(negative_sampling_distribution, dtype=np.int16)# , negative_sampling_probabilities, labels


class Word2Vec_CBOW(nn.Module):

    def __init__(self, embedding_size, n_inputs, n_outputs):
        super(Word2Vec_CBOW, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.embedding_size = embedding_size

        # Build layers
        self.embedding = nn.Embedding(num_embeddings=n_inputs,
                                      embedding_dim=embedding_size)
        
        self.linear = nn.Linear(in_features=embedding_size,
                                 out_features=n_outputs,
                                 bias=False)


    def forward(self, x):
        # u:     [batch_size, n_context,     embedding_size]
        embedding = self.embedding(x).sum(dim=1) # sum across all context words  
        # embedding = self.embedding(x).mean(dim=1) # mean across all context words  
        out = self.linear(embedding)
        # out = F.log_softmax(out, dim=1)
        return out

class Word2Vec_SkipGram(nn.Module):

    def __init__(self, embedding_size, n_inputs, n_outputs):
        super(Word2Vec_SkipGram, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.embedding_size = embedding_size

        # Build layers
        self.input_embeddings = nn.Embedding(n_inputs, embedding_size)
        self.output_embeddings = nn.Embedding(n_outputs, embedding_size)

    def init_embeddings(self):
        """Initialize embedding weight like word2vec.
        The u_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size], and the elements of v_embedding are zeroes.
        Returns:
            None
        """
        initrange = 0.5 / self.embedding_size
        self.input_embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_embeddings.weight.data.uniform_(-0, 0)
        # self.output_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, u, v_pos, v_neg):
        '''Parameters:
        u       - target input word
        v_pos   - positive context words
        v_neg   - non-context words (negative sampling)
        '''
        # u:     [batch_size, 1,     embedding_size]
        # v_pos: [batch_size, n_pos, embedding_size]
        # v_neg: [batch_size, n_neg, embedding_size]
        u = self.input_embeddings(u)
        v_pos = self.output_embeddings(v_pos)
        v_neg = self.output_embeddings(v_neg)

        # positive examples
        pos_score = torch.mul(u, v_pos)
        # summing across the embedding dimension gives us the dot product
        pos_score = torch.sum(pos_score, dim=-1)
        pos_score = F.logsigmoid(pos_score)

        # negative examples
        neg_score = torch.mul(u, v_neg)
        neg_score = torch.sum(neg_score, dim=-1)
        neg_score = F.logsigmoid(-1 * neg_score) # negate

        # combine positive and negative examples into loss
        return -1 * (torch.sum(pos_score)+torch.sum(neg_score))
