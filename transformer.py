import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn

'''
Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, decoder, inp_embed, generator, pos_embed):
        super(EncoderDecoder, self).__init__()
        # self.encoder = encoder
        self.decoder = decoder
        self.inp_embed = inp_embed
        self.generator = generator
        self.pos_embed = pos_embed
        
    def forward(self, inp, inp_mask, target_pos=None):
        "Take in and process masked src and target sequences."
        return self.generator(self.decode(inp, inp_mask, target_pos))
    
    def decode(self, inp, inp_mask, target_pos):
        inp = self.inp_embed(inp)
        if target_pos is not None:
            target_pos = self.pos_embed(inp, pos_only=True, trg_pos=target_pos)
        return self.decoder(inp, inp_mask, target_pos)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, inp_mask, target_pos_embeddings):
        for layer in self.layers:
            x = layer(x, inp_mask, target_pos_embeddings=target_pos_embeddings)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) 
 
    def forward(self, x, inp_mask, target_pos_embeddings=None):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, inp_mask, trg_pos_embed=target_pos_embeddings))
        if target_pos_embeddings is not None:
            x += target_pos_embeddings

        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, trg_pos_embed=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x, pos_only=False, trg_pos=None):
        pos = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        if not pos_only:
            x = x + pos
        else:
            assert trg_pos is not None
            trg_pos_embed = torch.index_select(pos, 1, trg_pos.flatten())
            trg_pos_embed = trg_pos_embed.view(trg_pos.size(0), trg_pos.size(1), pos.size(-1))
            x = trg_pos_embed
        return self.dropout(x)



def make_model(tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Decoder(DecoderLayer(d_model, c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab), c(position))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

# Layers
# ------------------------------------------------------------------------------------
''' Neat way of doing  ResNet while changing the dimension of the representation'''
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(h)

        g = self.sigmoid(self.g(x))

        return h * g



# Sampling
# ------------------------------------------------------------------------------------

def greedy_decode(model, max_len, start_symbol=2, take_max=False):
    ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()
    for i in range(max_len-1):
        out = model.decode(
                           ys,
                           (subsequent_mask(ys.size(1)).type_as(ys)), target_pos=None)
        prob = model.generator(out[:, -1])
        if take_max:
            _, next_word = torch.max(prob, dim = 1)
        else:
            dist = torch.distributions.Categorical(prob.exp())
            next_word = dist.sample()
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
        ys = ys.cuda()
    return ys


def low_entropy_decoding(model, max_len, sos_token, pad_token):
    ys = torch.ones(1, max_len).fill_(pad_token).long().cuda()
    ys[0, 0] = sos_token

    mask = torch.zeros(1, max_len, max_len).byte().cuda()
    
    # all tokens can look at the sos token
    mask[:, :, 0] = 1
  
    target_pos = torch.arange(max_len).unsqueeze(0).long().cuda()

    for t in range(max_len):
        out  = model.decode(ys, mask, target_pos)
        prob = model.generator(out)
        dist = torch.distributions.Categorical(prob.squeeze().exp())

        entropies = dist.entropy()                
        # zero-out words that have already been generated
        mask_t = (ys != pad_token).squeeze()
        entropies.masked_fill_(mask_t, 999999)

        position = entropies.argmin()
        sample = dist.sample()[position]

        # update the mask to take into account this new word
        mask[:, :, position] = 1
        ys[:, position] = sample
        
    return ys
    
