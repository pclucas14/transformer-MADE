import os
import torchtext
import numpy as np
from torchtext import data

class CustomDataset(data.Dataset):
    def __init__(self, fields, path=None, examples=None, **kwargs):
        if examples is None:
            examples = []
            lines = open(path, 'r').read().splitlines()
            lens = []
            for line in lines:
                parts = ['<sos>'] + line.strip().split(" ") #+ ['<eos>']
                lens += [len(parts)]
                examples += [data.Example.fromlist([parts], fields)]
                
            sorted_ind = np.argsort(lens)
            examples = [examples[i] for i in sorted_ind]
        super(CustomDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, train_file, valid_file, test_file, **kwargs):
        train_set = cls(fields, train_file, **kwargs)
        valid_set = cls(fields, valid_file, **kwargs)
        test_set  = cls(fields, test_file,  **kwargs)
        return train_set, valid_set, test_set


def to_readable(vocab, matrix):
    if isinstance(vocab, torchtext.data.field.Field):
        vocab = vocab.vocab.itos

    sentences = []
    for line in matrix:
        sentence = ''
        for token in line:
            sentence += vocab[token] + ' '
        sentences += [sentence]
    return sentences


# build dataset 
def load_data(train_file='train.txt', valid_file='valid.txt', test_file='test.txt', path=None, batch_size=256, **kwargs):
    if path is not None:
        train_file, valid_file, test_file = \
		[os.path.join(path, ext) for ext in [train_file, valid_file, test_file]]

    # create required field for language modeling
    input_field = data.Field(lower=True, batch_first=True)
    fields = [("text", input_field)]

    train_set, valid_set, test_set  = CustomDataset.splits(fields, train_file, valid_file, test_file)
    input_field.build_vocab(train_set)

    import pdb; pdb.set_trace()

    train_loader, val_loader, test_loader = data.Iterator.splits(
      (train_set, valid_set, test_set),
      sort_key=lambda x : len(x.text),
      batch_sizes=(batch_size, 512, 512),
      **kwargs)

    return input_field, train_loader, val_loader, test_loader
