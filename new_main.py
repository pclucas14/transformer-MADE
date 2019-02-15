import comet_ml
import os
import sys
import pdb
import torch
import numpy as np
from pydoc import locate
from comet_ml import Experiment
from torchtext import data, datasets
from collections import OrderedDict as OD
import matplotlib.pyplot as plti

from transformer import * 
from models      import * 
from utils       import * 
from custom_ds   import *
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

input_field, train_iter, val_iter, test_iter = load_data(path=args.data_dir, batch_size=args.batch_size)
iterators = {'train': train_iter, 'valid': val_iter, 'test': test_iter}

VOCAB = input_field.vocab.stoi
VOCAB_SIZE, PAD, SOS = len(VOCAB.keys()), VOCAB['<pad>'], VOCAB['<sos>']
best_valid = 1e10

# create model and ship to GPU
if args.model == 'transformer':
    gen = make_model(VOCAB_SIZE, N=args.n_layers, h=args.n_heads, dropout=0., d_ff=2048).cuda()
else:
    gen = RNNModel(args.model, VOCAB_SIZE, 650, 650, 650, 1).cuda()

lr = args.lr
optimizer_gen = locate('torch.optim.%s.' % args.optim)(gen.parameters(), lr=lr)
print('number of params', sum([np.prod([int(y) for y in x.shape]) for x in gen.parameters()]))
print(gen)


def full_epoch(epoch_no, split, masking):
    loader = iterators[split]

    # create logging containers
    logs = OD()
    for name in ['nll', 'ppl']:
        logs[name] = []

    gen.train() if split == 'train' else gen.eval()

    # Training loop
    for i, minibatch in enumerate(loader):
        text = minibatch.text.cuda()
        input = text

        #masking = ('left_to_right' if np.random.rand() > 0.2 else 'random') #if split == 'train' else masking

        if args.model == 'transformer':
            lens = (text != 1).sum(dim=1).byte().cpu().data.numpy()
            masks, orders, target_pos   = \
                [torch.from_numpy(x).long().cuda() 
                        for x in 
                 build_ar_masks(lens, order=masking)]

            offset = (torch.arange(orders.size(0)) * (orders.size(1))).unsqueeze(1).long().cuda()
            new_order = torch.take(input,  offset + target_pos)
            target = new_order

            # one token per sentence does not have a target anymore, here is the mask to mask it out
            mask_eos = target_pos   != -1 # -1 was used to denote the target of the last token
            
            # TODO: clean masking arg handling
            if args.mask_pad_tokens:
                mask_pad = text != PAD
                mask = torch.min(mask_eos, mask_pad)
            else:
                mask = mask_eos
        
            # turn -1s into 0s. They will be masked out during loss calculation
            target_pos = target_pos.clamp(min=0)
            
            # TODO: add some target information
            logits = gen(input, masks, target_pos=target_pos if masking=='random' or True else None) 
        
        else:
            # TODO: make LSTM implementation batch first
            # reverse input
            idx = torch.cuda.LongTensor([i for i in range(input.size(1)-1, -1, -1)])
            input = input.index_select(1, idx)
            input = input.transpose(1,0).contiguous()
            
            input, target = input[:-1], input[1:]
            logits = gen(input)[0]
            mask = target != PAD if args.mask_pad_tokens else torch.ones_like(target)

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
    
    if args.model == 'transformer':
        # left to right sampling
        left_to_right = greedy_decode(gen, 51, start_symbol=SOS)
        ltr_str = to_readable(input_field, left_to_right)[0]
        print('left to right : %s' % ltr_str)

        # low entropy sampling
        #low_entropy = low_entropy_decoding(gen, 51, SOS, PAD)
        le_str = ''#to_readable(input_field, low_entropy)[0]
    else:
        sample = gen.sample(51, SOS, 1).transpose(1,0)
        ltr_str = to_readable(input_field, sample)[0]
        print('left to right : %s' % ltr_str)
        le_str = ''

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
