import pdb
from pydoc import locate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence as KL


# Models
# ----------------------------------------------------------------------------------

""" RNN that supports Variational Dropout. Heavily copied from AWD's RNNModel """
class RNNModel(nn.Module):
    def __init__(self, rnn, vocab_size, input_size, hidden_size, output_size, num_layers, dropouts=[0.1, 0.5, 0.5, 0.5]):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()

        # dropouts ordered as they are used : embedding, input, hidden, output
        self.edrop, self.idrop, self.hdrop, self.odrop = dropouts
        self.embedding = nn.Embedding(vocab_size, input_size)
        rnn = locate('torch.nn.%s.' % rnn)

        self.rnns = nn.ModuleList([ rnn(input_size  if l == 0 else hidden_size, 
                                    hidden_size if l != num_layers - 1 else output_size, 1)
                                    for l in range(num_layers) ])

        self.out = nn.Linear(output_size, vocab_size)
        self.out.weight = self.embedding.weight

    def embed(self, x):
        if 'FloatTensor' in x.type(): return x
        emb = embedded_dropout(self.embedding, x, dropout=self.edrop if self.training else 0)
        return self.lockdrop(emb, self.idrop)
    
    def forward(self, x, hidden=None, step=None):
        if not isinstance(hidden, list):
            hidden = [hidden] * len(self.rnns)

        x = self.embed(x)
        raw_outputs, new_hs = [], []
        raw_output = x
        
        for l, rnn in enumerate(self.rnns):
            input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            
            # store appropriate info
            raw_outputs += [raw_output]
            new_hs += [new_h]

            if l != len(self.rnns) - 1:
                raw_output = self.lockdrop(raw_output, self.hdrop, step=step)
        
        output = self.lockdrop(raw_output, self.odrop, step=step)
        
        # output layer
        output = self.out(output)
    
        return output, new_hs

    
    def sample(self, max_len, sos_token, num_samples=100, temp=1):
        sos_token = torch.cuda.LongTensor(num_samples).fill_(sos_token)
        dec_hidden, prior_hidden = None, None
        words = []

        x_idx = sos_token
        for t in range(max_len):
            dec_input = self.embedding(x_idx).unsqueeze(0)
            
            logits, dec_hidden = self.forward(dec_input, dec_hidden)
            dist = logits.squeeze(0) * temp
            x_idx = Categorical(logits=dist).sample()
            words += [x_idx]
        
        return torch.stack(words, dim=0)

# Layers 
# ----------------------------------------------------------------------------------

""" Variational Dropout. Taken from https://github.com/salesforce/awd-lstm-lm """
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5, step=None):
        if not self.training or not dropout:
            return x
        
        # reset mask on first step
        if not step: 
            self.mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            self.mask = self.mask / (1 - dropout)
            self.mask.requires_grad = False
            self.mask = self.mask.expand_as(x)
        
        return self.mask * x


""" Embedded Dropout. Taken from https://github.com/salesforce/awd-lstm-lm """
def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1))\
                .bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = 2 # specific to our repo

    X = F.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse)

    return X
