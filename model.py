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
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, "d_model is not divissble by num_heads"

        self.d_k = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv
        
        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # calculating attention
        # (Batch, num_h, seq_len, d_k) --> (Batch, num_h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        # applying mask before softmax
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, num_h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores



    def forward(self, q, k, v, mask): # if we want some words not interacting with each other mask is used
        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.q_k(k)    # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.q_v(v)   # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, num_h, d_k) --> (Batch, num_h, seq_len, d_k)
        # transpose if done to make sure that each head sees the entire sentence but small parts of the embeddings
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, self.dropout)

        # (Batch, num_head, seq_len, d_k) --> (Batch, seq_len, num_head, d_k) --> (Batch, seq_len, d_model)
        # using contiguous so pytorch would do it in-place for memory management
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# turning to the decoder
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_froward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_froward_block = feed_froward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_froward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(mm.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed:Input_embedding, tgt_embed: Input_embedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer=ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    



    
        