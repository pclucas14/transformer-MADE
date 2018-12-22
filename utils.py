import time
import torch
import random
import threading
import torchtext
import numpy as np 

# -----------------------------------------------------------------------------------------------
# Logging 
# -----------------------------------------------------------------------------------------------

def print_scalar(name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))


def to_comet(value_dict, prefix=''):
    out = {}
    for key, value in value_dict.items():
        if len(value) > 0:
            value = torch.stack(value).mean().item()
            out[prefix + key] = value

    return out

