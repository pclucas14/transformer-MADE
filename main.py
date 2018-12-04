import os
import pdb
import numpy as np
import torch
from transformer import * 
from utils import * 
from torchtext import data, datasets
from collections import OrderedDict as OD

# enable random autoregressive masks
USE_RANDOM_AR_MASKS = True


# reproducibility
torch.manual_seed(2)
np.random.seed(2)

# build dataset 
train_iter, val_iter, test_iter = datasets.PennTreebank.iters(batch_size=512)
iterators = {'train': train_iter, 'valid': val_iter, 'test': test_iter}

writes = 0
best_valid = 1e5

# create model and ship to GPU
gen  = make_model(10000 + 1, N=2, h=4).cuda()
print(gen)
print('number of params', sum([np.prod([int(y) for y in x.shape]) for x in gen.parameters()]))

# build optimizer
optimizer_gen = torch.optim.Adam(gen.parameters())


def full_epoch(epoch_no, split):
    loader = iterators[split]

    # create logging containers
    logs = OD()
    for name in ['nll', 'ppl']:
        logs[name] = []

    gen.train() if split == 'train' else gen.eval()

    # Training loop
    for i, minibatch in enumerate(loader):
       
        input  = minibatch.text.transpose(1,0).cuda()
        target = minibatch.target.transpose(1,0).cuda()
        
        bs, seq_len = input.size()
        masks = torch.from_numpy(build_ar_masks([seq_len] * bs)).long().cuda()
        
        if i == 0:
            # import pdb; pdb.set_trace()
            print(masks[0])

        logits = gen(input, masks)
        recon_loss = F.cross_entropy(logits.view(bs * seq_len, -1), target.flatten())

        if gen.training:
             optimizer_gen.zero_grad()
             recon_loss.backward()
             params = optimizer_gen.param_groups[0]['params']
             torch.nn.utils.clip_grad_norm_(params, 10, norm_type=2)
             optimizer_gen.step()
         
        #import pdb; pdb.set_trace()
        #test = greedy_decode(gen, args.max_seq_len) 

        logs['nll']  += [recon_loss.data]
        logs['ppl']  += [recon_loss.exp().data]

    return logs


# Train / Test Loop
# -------------------------------------------------------------------------------------
for epoch in range(100):
    # print('MLE pretraining epoch {}/{}'.format(epoch, args.epochs))
    train_log = full_epoch(epoch, 'train')

    for key, value in train_log.items():
        print_scalar('train/%s' % key, value, writes)

    print('')
    if True: #(epoch + 1) % args.test_every == 0:
        with torch.no_grad():
            valid_log = full_epoch(epoch, 'valid')

            for key, value in valid_log.items():
                print_scalar('valid/%s' % key, value, writes)
            
            # keep tab of best valid error in order to get legit test error:
            curr_valid_loss = torch.stack(valid_log['nll']).mean().item()
            if curr_valid_loss < best_valid: 
                best_valid = curr_valid_loss
                best_valid_epoch = epoch
                torch.save(gen.state_dict(), 'models/gen.pth')
            else:
                pass
                # lr /= args.lr_div
                # for pg in optimizer_gen.param_groups: pg['lr'] = lr
                # print('\n reducing lr to {} \n'.format(lr))
                # end prematurely if learning rate is super small
                # if lr < 1e-10 : break
            
    writes += 1

""" Testing Best Model """
print('loading model with best valid loss')
gen.load_state_dict(torch.load('models/gen.pth'))
test_log = full_epoch(epoch, 'test')

for key, value in test_log.items():
    print_scalar('test/%s' % key, value, 0)
