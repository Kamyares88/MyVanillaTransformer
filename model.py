import torch
import torch.nn as nn
import math


# 1. writing the input embedding section
class Input_embedding(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        '''
        d_model: model dimension, in this case 512, it is the size of input embedding
        vocab_size: the size of the vocabulary, number of words (tokens) in the vocabulary
        '''
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# 2. positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # create a matric of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # making a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad(False) # we don't want the PE to be learnable
        return self.dropout(x)

# layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # learnable
        self.bias = nn.Parameter(torch.ones(1)) # learnable

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x-mean)/(std + self.epsilon) + self.bias
    
# feed forward layer; info from paper:
# FFN(x) = max(0, xW1+b1)W2 + b2 --> these can be modeled as two densenet
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# Multi-head attention layer


    
        