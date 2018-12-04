import torch
import numpy as np 


def build_ar_masks(lens):
    masks = []
    for len_ in lens:
        arr = np.arange(len_)
        np.random.shuffle(arr)
        
        rev = [arr[arr[j]] for j in range(len_)]
        mask = np.zeros((len_, len_))
        for j, row in enumerate(mask[:len_]): 
            # row[i] = 1
            # find index with i
            index = np.where(arr == j)[0][0]
            row[arr[:index]] = 1

        mask = mask + np.eye(mask.shape[0])
        mask[len_:, len_:] = 0

        masks += [mask]

    return np.stack(masks)
    
def print_scalar(name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))

    