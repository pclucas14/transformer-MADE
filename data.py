import re
import os
import time
import torch
import random
import threading
import torchtext
import numpy as np 
import _pickle as pickle

# -----------------------------------------------------------------------------------------------
# Data Preprocessing 
# -----------------------------------------------------------------------------------------------
# needs to tokenize and give a dict. Needs also to allow to load pickled objects

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_set = set()

        # add <unk> <sos> and <eos> tokens
        # really important not to change (hardcoded in minibatch_generator)
        self.add_word(u'<pad>')  # ID 0
        self.add_word(u'<eos>')  # ID 1
        self.add_word(u'<sos>')  # ID 2
        self.add_word(u'<unk>')  # ID 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def to_readable(self, matrix):
        sentences = []
        for line in matrix:
            sentence = ''
            for token in line:
                sentence += self.idx2word[token] + ' '
            sentences += [sentence]
        return sentences


def tokenize(path, train=False, word_dict=None, dataset=None):
    # tokenizing process is somewhat lenghty. Let's try to avoid it when possible
    try:
        path_word_dict = path + '_word_dict.pickle'
        path_ids = path + '_ids.pickle'
        with open(path_ids, 'rb') as f: 
            ids = pickle.load(f)
        if train: 
            with open(path_word_dict, 'rb') as f: 
                word_dict = pickle.load(f)
        
        print('loaded preprocessed data from %s' % path)
        return ids, word_dict
    except: 
        pass 

    """Tokenizes a text file."""
    if word_dict is None : 
        print('creating new word dictionary')
        word_dict = Dictionary() 
    assert os.path.exists(path), '{} does not exist'.format(path)
    # Add words to the dictionary
    with open(path, 'r') as f:
        sentences = 0
        max_tokens = 0
        for line in f:
            # line = line.decode('utf-8', 'strict')
            words = re.findall(r"[\w']+|[.,!?;]", line,
                    flags=re.UNICODE) 
            
            if words[-1] == '.':
                words[-1] = '<eos>'
            elif words[-1] == '?':
                words[-1] =  '<qm>'
            elif words[-1] == '!':
                words[-1]  ='<em>'

            if 'ptb' in dataset:
                words += ['<eos>']
        
            # only add words if in training set
            if train:
                for word in words:
                    word_dict.add_word(word)
                word_dict.vocab_set = \
                    set(word_dict.idx2word)

            # track stats for building tokenized version
            tokens = len(words)
            sentences += 1
            if tokens > max_tokens:
                max_tokens = tokens

    # Tokenize file content
    with open(path, 'r') as f:
        ids = []
        for i, line in enumerate(f):
            # line = line.decode('utf-8', 'strict')
            words = re.findall(r"[\w']+|[.,!?;]", line, 
                    flags=re.UNICODE)
            
            if words[-1] == '.':
                words[-1] = '<eos>'
            elif words[-1] == '?':
                words[-1] =  '<qm>'
            elif words[-1] == '!':
                words[-1]  ='<em>'

            if 'ptb' in dataset:
                words += ['<eos>']
            
            token = 0
            idx = list(range(len(words)))
            for word in words:
                if word not in word_dict.vocab_set:
                    word = u'<unk>'
                idx[token] = word_dict.word2idx[word]
                token += 1
            
            # add <SOS>
            idx = [word_dict.word2idx['<sos>']] + idx

            # create list of lists for easier process later on
            ids.append(idx)

    # save to file 
    path_word_dict = path + '_word_dict.pickle'
    path_ids = path + '_ids.pickle'
    with open(path_ids, 'wb') as f: 
        pickle.dump(ids, f)
    if train: 
        with open(path_word_dict, 'wb') as f: 
            pickle.dump(word_dict, f)
    
    return ids, word_dict


# -----------------------------------------------------------------------------------------------
# Utilities for data handling and target creation 
# -----------------------------------------------------------------------------------------------

def build_ar_masks(lens, order='random', verbose=False):
    """ create autoregressive masks with arbitrary orderings + other useful tensors """ 
    assert order in ['random', 'left_to_right']
    max_len = max(lens)
    
    # placeholder for (previous) arbitrary word ordering
    masks, orders, targets  = [], [], []

    # placeholder for flexible positional decoding
    targets_words, targets_pos, tfs_bin_pos = [], [], []

    for len_ in lens:
        arr = np.arange(len_)
        
        if order == 'random':
            arr = np.arange(1, len_-1)
            np.random.shuffle(arr)
            
            # SOS always first, EOS always last
            arr = [0] + list(arr) + [len_ - 1]

        arr = np.concatenate([arr, np.arange(len_, max_len)])
        orders += [arr]
        target = []
        mask = np.zeros((max_len, max_len))

        # placeholder for flexible positional decoding
        target_words = []
        target_pos   = []
        tf_bin_pos   = [] # teacher forced positions

        for j, row in enumerate(mask[:len_-1]):
            # find position of x_{t==j} in the ordering
            index = np.where(arr == j)[0][0]

            # for x_{t==j}, the model can `see` tokens before it according
            # to the ordering prescribed in `arr`
            row[arr[:index]] = 1

            # all tokens but the last have a target
            target_value = arr[index+1]
            target += [target_value]

            # possible positions for next token. Note that we restrain the model
            # from going to the left of SOS token (which is always first)
            if index == 0:
                possible_pos = [len_ - 1]
                target_bin   = 0
                pos_in_target_bin = [x for x in range(1, len_)]
            else:
                possible_pos = np.zeros((index + 1))
                curr_ind = 0
                target_bin = None
                pos_in_target_bin = []
                target_pos_found = False

                # we iterate over the row to find the bins, and the candidate tokens
                # for the bin index that will be teacher forced.
                for k in range(1, len_):  
                    if row[k] == 1 or k == j:
                        curr_ind += 1
                        if target_bin is None:
                            pos_in_target_bin = []
                        else:
                            target_pos_found = True
                    else:
                        possible_pos[curr_ind] += 1
                        
                        if not target_pos_found:
                            pos_in_target_bin += [k]

                    if k == target_value:
                        target_bin = curr_ind
            
            # since <eos> is always last, we let it condition on all tokens
            mask[len_-1] = np.ones_like(mask[len_-1])
            # we add np.eye() later on
            mask[len_-1, len_-1] = 0

            # during training, we teacher force the position determined by 
            # the ordering in `arr`

            target_words += [pos_in_target_bin]
            target_pos   += [possible_pos]
            tf_bin_pos   += [target_bin]

        # we set the target to `-2` for pad tokens
        target += [-2] * (max_len - len(target))
        mask = mask + np.eye(mask.shape[0])
        mask[len_:, len_:] = 0
        
        # store sentence info (for arbitrary word orderings)
        masks       += [mask]
        targets     += [target]
        targets_pos += [target_pos]

        # store sentence info (for flexible positional decoding)
        targets_words += [target_words]
        tfs_bin_pos   += [tf_bin_pos]

        if verbose:
            sentence = [chr(ord('a') + i) for i in range(len_-2)]
            sentence = ['<sos>'] + sentence + ['<eos>']
            print('sentence : ', sentence)
            print('order    : ', arr, '\n')

            for i in range(len_ -1): # no word target for <eos> token
                ind = np.where(arr == i)[0][0]
                gen = ''
                for token in sorted(arr[:ind+1]):
                    gen += str(sentence[token]) + ' _ '
               
                print('curr pos   ', i) 
                print('target pos ', arr[ind+1], ' --> ', sentence[arr[ind+1]])
                print('generated  ', gen if len(gen) > 0 else '_ ')
                print('bins       ', target_pos[i])
                print('tf bin     ', tf_bin_pos[i])
                print('bin words  ', [sentence[x] for x in target_words[i]])
                print('')
            
            print(mask)
            print('\n\n')

    masks, orders, targets = [np.stack(x) for x in [masks, orders, targets]]

    return masks,           orders,       targets,\
           targets_words,   targets_pos,  tfs_bin_pos


def build_target_dists(data, target_words, target_bins, args):
    """ converts bins to actually torch.Distributions """ 
    
    # for efficiency, return (batched) distributions of shape (bs * seq_len, vocab_size)
    bs, seq_len = data.size()
    num_pos     = args.max_seq_len + 1

    # placeholders. init to uniform, so that pytorch does not complain. 
    # note that distributions that are not overwritten will be masked out (e.g. for PAD tokens)
    words_prob = torch.ones(bs, seq_len, args.vocab_size) / args.vocab_size
    bins_prob  = torch.ones(bs, seq_len, num_pos)         / num_pos
    
    num_sentence = data.shape[0]
    for i, sent  in enumerate(data):
        tgt_words = target_words[i]
        tgt_bins  = target_bins[i]
        seq_len_m_1  = len(tgt_words[0])

        if len(tgt_words) != len(tgt_bins):
            print(len(tgt_words), len(tgt_bins))

        for t in range(seq_len_m_1):
            tgt_word = tgt_words[t]
            tgt_bin  = tgt_bins[t]
            
            # 1) build positional target
            if args.pos_embeddings == 'relative':
                raise NotImplementedError
            elif args.pos_embeddings == 'absolute':
                bins_prob[i, t] = 0.
                bins_prob[i, t, :len(tgt_bin)] = torch.Tensor(tgt_bin)

            # 2) build word target
            # tgt_words contains the positions, but we need the word indices
            tgt_indices = data[i, tgt_word]
            words_prob[i, t] = 0.
            words_prob[i, t, tgt_indices] = 1.
                
            # normalize
            words_prob[i, t] /= words_prob[i, t].sum()
            bins_prob[i, t] /= bins_prob[i, t].sum()

        # for last token, target is the <end sentence> positional token
        # which is the last token 
        bins_prob[i, seq_len_m_1] = 0.
        bins_prob[i, seq_len_m_1, -1] = 1.

    if args.cuda: 
        words_prob = words_prob.cuda()
        bins_prob  = bins_prob.cuda()

    # all probs are calculated. wrap things up in a distribution
    words_dist = torch.distributions.Categorical(words_prob.view(-1, args.vocab_size))
    bins_dist  = torch.distributions.Categorical(bins_prob.view(-1, num_pos))

    return words_dist, bins_dist
            

def fill_seq(input, padded_length, fill_token):
    input_padded = input[:padded_length]
    input_padded += [fill_token] * (padded_length - len(input))
    return input_padded


# -----------------------------------------------------------------------------------------------
# Multi-threaded Data Iterators  
# -----------------------------------------------------------------------------------------------

class DataLoader():
    def __init__(self, dataset, args, vocab, shuffle=True, order='random'):
        self.ds = dataset
        self.args = args
        self.next_batch_ready = False
        self.out = None
        self.vocab = vocab

        # Assumes SOS and EOS tokens have already been added.
        self.PAD_token = vocab.word2idx['<pad>'] 
        self.SOS_token = vocab.word2idx['<sos>'] 

        # shuffle dataset
        random.shuffle(dataset)

        # then sort by length for efficiency
        dataset.sort(key=len)

        # divide into chunks of length `args.batch_size`
        minibatches = [ dataset[i:i + args.batch_size] for i in range(0, len(dataset), args.batch_size) ]

        if args.drop_last: 
            minibatches = minibatches[:-1]

        self.minibatches = minibatches
        self.order       = order
        self.shuffle     = shuffle

        self.reset_indices()
        self.reset_thread()


    def get_next_batch(self):
        """ method called by thread to prepare new batch """ 
        args = self.args
        ind = self.indices.pop()
        self.num_batches -= 1

        minibatch = self.minibatches[ind]

        # longest sentence always at the end of the minibatch
        max_len   = min(args.max_seq_len, len(minibatch[-1]))
        minibatch = [fill_seq(sent, max_len, self.PAD_token) for sent in minibatch]
        data      = torch.LongTensor(minibatch)
        
        # collect lengths to build masks
        lens = [min(len(x), args.max_seq_len) for x in minibatch]

        # build mask + stuff required to build target
        mask, order, target_i, target_words, target_pos, tf_pos = build_ar_masks(lens, order=self.order)
        mask, order, target_i = [torch.LongTensor(x) for x in [mask, order, target_i]]

        if args.cuda:
            data, mask, order, target_i = data.cuda(), mask.cuda(), order.cuda(), target_i.cuda()

        if args.mode == 'PTW':
            words_dist, bin_dist = build_target_dists(data, target_words, target_pos, args)
            out = data, words_dist, bin_dist, mask # do we need something else ?
        else:
            out = data, mask, order, target_i
        
        self.out = out
        self.next_batch_ready = True 


    def reset_thread(self):
        """ creates and starts new thread """
        if self.args.debug:
            self.get_next_batch()
        else:
            self.thread = threading.Thread(target = self.get_next_batch, args=()) 
            self.thread.start()


    def reset_indices(self):
        """ resets the interation process over the minibatches """ 
        self.num_batches = len(self.minibatches)
        self.indices = list(np.arange(self.num_batches))

        if self.shuffle: 
            random.shuffle(self.indices)


    def new_epoch(self):
        """ handles all the resetting when starting new epoch """ 
        self.reset_indices()
        self.reset_thread()

    
    def __iter__(self):
        """ iterator method """ 
        return self


    def __next__(self):
        """ iterator method """ 
        waited = 0
        while not self.next_batch_ready : 
            waited += 1
            time.sleep(1)
            print("waited {} seconds for next batch".format(str(waited)))

        out = self.out
        if self.num_batches <= 0: 
            self.new_epoch()
            raise StopIteration

        self.reset_thread()
        return out
        

if __name__ == '__main__':
    class args:
        pass

    args.cuda = True
    args.mask_pad_tokens = True
    args.batch_size = 256
    args.max_seq_len = 51
    args.drop_last = False
    args.mode = 'PTW' # Position, Then Word
    args.vocab_size = 5000
    args.num_pos  = 60
    args.pos_embeddings = 'absolute'

    data = [[ np.random.randint(args.vocab_size) for _ in range(np.random.randint(40, 50)) ] for _ in range(1000) ]

    dataloader = DataLoader(data, args)
    start = time.time()    

    for i, out in enumerate(dataloader):
        i += 1
        print(i)
        # do some stuff for a second
        time.sleep(1)

    end = time.time()
    delta = end - start
    print('took {:.4f} with threading, and overhead {:.4f}'.format(delta, delta - i))
