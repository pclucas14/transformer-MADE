import comet_ml
import os
import sys
import pdb
import time
import torch
import numpy as np
from pydoc import locate
from comet_ml import Experiment
from torchtext import data, datasets
from collections import OrderedDict as OD

from transformer import * 
from models      import * 
from utils       import * 
from data        import * 
from args        import * 

# ----------------------------------------------------------------------------------------------
# Main file. Trying a bunch of masking techniques 
# ----------------------------------------------------------------------------------------------

# reproducibility
torch.manual_seed(2)
np.random.seed(2)
args = get_args()
print(args)

# Build Experiment Container
if args.log:
    experiment = Experiment(api_key="WXSjTPlJVTQlUBN2o3O5D6Pwz", 
                            project_name="transformer-made", 
                            workspace="pclucas14")
    experiment.log_parameters(vars(args))

# Tokenize data
dataset_train, word_dict = tokenize(os.path.join(args.data_dir, 'train.txt'), \
        train=True, dataset=args.data_dir)
dataset_valid, word_dict = tokenize(os.path.join(args.data_dir, 'valid.txt'), train=False, \
        word_dict=word_dict, dataset=args.data_dir)
dataset_test,  word_dict = tokenize(os.path.join(args.data_dir, 'test.txt'),  train=False, \
        word_dict=word_dict, dataset=args.data_dir)
args.vocab_size = len(word_dict) 

# Build torch.Dataset
datasets = {'train': TextDataset(dataset_train, args, word_dict, order=args.masking),
            'valid': TextDataset(dataset_valid, args, word_dict, order=args.masking), 
            'test':  TextDataset(dataset_test,  args, word_dict, order=args.masking)}

def collate_fn(data):
    out = []
    for item in data[0]:
        if type(item) == torch.Tensor: item = item.squeeze(0)
        out += [item]
    return out

# Build torch.DataLoaders
loader = lambda x : torch.utils.data.DataLoader(x, shuffle=args.shuffle, 
                num_workers=args.n_workers, collate_fn=collate_fn)
iterators = {ds_name:loader(ds) for (ds_name, ds) in datasets.items()}

# useful quantities
lr = args.lr
best_valid = 1e10
SOS, EOS, PAD = [word_dict.word2idx[x] for x in ['<sos>', '<eos>', '<pad>']]

# create model and ship to GPU
if args.model == 'transformer':
    gen = Transformer(args).cuda()
else:
    gen = RNNModel(args.model, args.vocab_size, 650, 650, 650, 1).cuda()

# build optimizer
optimizer_gen = locate('torch.optim.%s.' % args.optim)(gen.parameters(), lr=lr)
print('number of params', sum([np.prod([int(y) for y in x.shape]) for x in gen.parameters()]))


def full_epoch(epoch_no, split, masking):
    loader = iterators[split]
    start  = time.time()

    # create logging containers
    logs = OD()
    for name in ['nll', 'ppl']:
        logs[name] = []

    gen.train() if split == 'train' else gen.eval()

    # Training loop
    for i, minibatch in enumerate(loader):
        if args.mode == 'RM': 
            if args.cuda: minibatch = [x.cuda() for x in minibatch]            
            input, masks, orders, target_pos = minibatch
            offset = (torch.arange(orders.size(0)) * (orders.size(1))).unsqueeze(1).long().cuda()
            total = (offset + target_pos).clamp_(min=0)
            new_order = torch.take(input,  total) #offset + target_pos.clamp(min=0))
            target = new_order
            tf_pos = None
            
            # turn -1s into 0s. They will be masked out during loss calculation
            target_pos = target_pos.clamp(min=0)
        
        elif args.mode == 'PTW':
            input, masks, orders, tf_pos, dist_pos, dist_word = minibatch
            # since last token is always eos, we don't need to predict anything for it
            bs, seq_len = input.size()
            input = input[:, :-1] 
            masks = masks[:, :-1, :-1]
            tf_pos = tf_pos[:, :-1]
            dist_pos.probs  = dist_pos.probs.view(bs, seq_len, -1)[:, :-1].reshape(bs * (seq_len-1), -1)
            dist_word.probs = dist_word.probs.view(bs, seq_len, -1)[:, :-1].reshape(bs * (seq_len-1), -1)
            
            if args.cuda:
                input, masks, orders, tf_pos = [x.cuda() for x in [input, masks, orders, tf_pos]]
                dist_pos.probs  = dist_pos.probs.cuda()
                dist_word.probs = dist_word.probs.cuda()

        if args.model == 'transformer':
            # one token per sentence does not have a target anymore, here is the mask to mask it out
            mask_eos = input != EOS
            
            # TODO: clean masking arg handling
            if args.mask_pad_tokens:
                mask_pad = input != PAD
                mask = torch.min(mask_eos, mask_pad)
            else:
                mask = mask_eos
        
            if args.mode == 'RM':
                # logits_word = gen(input, masks, target_pos=target_pos if masking=='random' or True else None)[0]
                logits = gen(input, masks, target_pos=target_pos if masking=='random' else None)
            elif args.mode == 'PTW':
                logits_word, logits_pos = gen(input, masks, tf_pos=tf_pos) 
                est_word, est_pos = [torch.distributions.Categorical(logits=x.view(bs * (seq_len-1), -1)) \
                                            for x in [logits_word, logits_pos]]
  
                mask = mask.float().flatten()
                kl_word = torch.distributions.kl.kl_divergence(dist_word, est_word) * mask
                kl_pos  = torch.distributions.kl.kl_divergence(dist_pos,  est_pos)  * mask
                recon_loss = (kl_word + kl_pos).sum()
        
        else:
            # TODO: make LSTM implementation batch first
            input = input.transpose(1,0).contiguous()
            
            input, target = input[:-1], input[1:]
            logits = gen(input)[0]
            mask = target != PAD if args.mask_pad_tokens else torch.ones_like(target)

        if args.mode != 'PTW':
            bs, seq_len = input.size()
            recon_loss = F.cross_entropy(logits.view(bs * seq_len, -1), target.flatten(), reduction='none')
            recon_loss = recon_loss.reshape(*input.size())
        
            # mask out the conditionals with no target and mask out the pad tokens
            recon_loss = recon_loss * mask.float()
            recon_loss = recon_loss.sum() / mask.sum().float()

        if gen.training:
             optimizer_gen.zero_grad()
             recon_loss.backward()
             torch.nn.utils.clip_grad_norm_(gen.parameters(), args.clip, norm_type=2)
             optimizer_gen.step()
         
        logs['nll']  += [recon_loss.data]
        logs['ppl']  += [recon_loss.exp().data]

    logs['time'] = time.time() - start
    return logs


# Train / Test Loop
# -------------------------------------------------------------------------------------
for epoch in range(args.epochs):
    print('masking : %s' % args.masking)
    
    train_log = full_epoch(epoch, 'train', masking=args.masking)

    for key, value in train_log.items():
        print_scalar('train/%s' % key, value, epoch)
    print('')
    
    with torch.no_grad():
        valid_log = full_epoch(epoch, 'valid', masking=args.masking)

        for key, value in valid_log.items():
            print_scalar('valid/%s' % key, value, epoch)
        print('')
    
    '''
    if args.model == 'transformer':
        # left to right sampling
        left_to_right = greedy_decode(gen, 51, start_symbol=SOS)
        ltr_str = word_dict.to_readable(left_to_right)[0]
        print('left to right : %s' % ltr_str)

        # low entropy sampling
        low_entropy = low_entropy_decoding(gen, 51, SOS, PAD)
        le_str = word_dict.to_readable(low_entropy)[0]
        print('low entropy : %s' % le_str)
    else:
        sample = gen.sample(51, SOS, 1).transpose(1,0)
        ltr_str = word_dict.to_readable(left_to_right)[0]
        print('left to right : %s' % ltr_str)
        le_str = ''
    '''

    if args.log:
        experiment.log_metrics(to_comet(train_log, prefix='train_'), step=epoch)
        experiment.log_metrics(to_comet(valid_log, prefix='valid_'), step=epoch)
        #experiment.log_other('left to right sample %d' % epoch, ltr_str)
        #experiment.log_other('low entropy sample %d' % epoch,   le_str)

    curr_valid_loss = torch.stack(valid_log['nll']).mean().item()
    if curr_valid_loss > best_valid:
        lr /= args.lr_div
        for pg in optimizer_gen.param_groups: pg['lr'] = lr
        print('\n reducing lr to {} \n'.format(lr))
        if args.log: experiment.log_metric('lr', lr, step=epoch)
        if lr < 1e-10 : sys.exit()
    else:
        best_valid = curr_valid_loss
