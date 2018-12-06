import os
import pdb
import copy
import numpy as np
import torch
from transformer import * 
from utils import * 
from collections import OrderedDict as OD

seq_len = 20
tries   = 300

masks = torch.from_numpy(build_ar_masks([seq_len ] * tries)).byte().cuda()

model = make_model(seq_len, N=2, h=4, d_model=100).cuda()
nn.init.normal_(model.tgt_embed[0].lut.weight, 2, 2)


for i in range(tries):
    print('%d / %d' % (i, tries))
    input = torch.arange(seq_len).unsqueeze(0).long().cuda()
    mask  = masks[[i]] # make_std_mask(input, 999) #masks[[i]]

    # -----------------------------------------------------------------------------------
    # We first test if the dependency given by the mask is enforced in the fwd/bwd pass
    # -----------------------------------------------------------------------------------

    for t in range(input.size(1)):
        model.zero_grad()
        out = model(input, mask) * 1000
        loss = out[:, t].sum().backward()
        test = model.tgt_embed[0].lut.weight.grad 
        depends = (model.tgt_embed[0].lut.weight.grad != 0).byte()
        depends = depends.sum(dim=1)
        depends = depends > 0
        
        if (depends != mask[0, t]).sum() > 0:
            raise ValueError('Mask dependency is not being enforced')
    
    # -----------------------------------------------------------------------------------
    # Next, we test if the mask is actually autoregressive
    # -----------------------------------------------------------------------------------
    
    mask = mask.squeeze()
    depends = []

    for row in mask:
        depends += [np.argwhere(row > 0)]
    
    # the bigger the dependency list, the further you are in the ordering
    order = np.argsort([x.size(1) for x in depends])

    if i % 100 == 0:
        for i in order:
            print(depends[i])

    for i  in range(1, len(order)):
        current_index = order[i]
        prev_deps = [int(x) for x in depends[order[i-1]].squeeze(0)]
        curr_deps = [int(x) for x in depends[order[i]].squeeze(0)]
        
        # previous list of dependency sould equal the current one, minus the self dependency
        prev_deps += [int(current_index)]
    
        if sorted(prev_deps) != sorted(curr_deps):
            raise ValueError('Ordering is breaking autoregressive property')

