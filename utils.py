import torch
import torchtext
import numpy as np 


def build_ar_masks(lens, order='random', num_swaps=None):
    assert order in ['random', 'left_to_right']
    max_len = max(lens)
    masks, orders, targets, distances = [], [], [], []
    for len_ in lens:
        arr = np.arange(len_)
        
        if order == 'random':
            if num_swaps is None:
                np.random.shuffle(arr)
                distances += [np.absolute(np.arange(len_) - arr).sum()]
            else:
                for _ in range(num_swaps):
                    s_ind = np.random.randint(len_)
                    t_ind = np.random.randint(len_)
                    temp   = arr[s_ind]
                    arr[s_ind] = arr[t_ind]
                    arr[t_ind] = temp
                    distances += [num_swaps]                 
        else:
            distances += [0]

        arr = np.concatenate([arr, np.arange(len_, max_len)])

        orders += [arr]

        rev = [arr[arr[j]] for j in range(len_)]
        target = []
        mask = np.zeros((max_len, max_len))
        for j, row in enumerate(mask[:len_]):
            # row[i] = 1
            # find index with i
            index = np.where(arr == j)[0][0]
            row[arr[:index]] = 1

            if index < len_ - 1: # not last
                target += [arr[index+1]]
            else:
                target += [-1]

        target += [-2] * (max_len - len(target))
        mask = mask + np.eye(mask.shape[0])
        mask[len_:, len_:] = 0

        masks += [mask]
        targets += [target]

    return np.stack(masks), np.stack(orders), np.stack(targets), np.array(distances)


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
