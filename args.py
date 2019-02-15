import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',        type=str,   default='data/ptb')
    parser.add_argument('--mask_pad_tokens', type=int,   default=1)
    parser.add_argument('--log',             type=int,   default=1)
    parser.add_argument('--masking',         type=str,   default='left_to_right', choices=['left_to_right', 'random'])
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--epochs',          type=int,   default=300)
    parser.add_argument('--model',           type=str,   default='transformer')
    parser.add_argument('--n_heads',         type=int,   default=2,    help='number of heads for transformer architecture')
    parser.add_argument('--n_layers',         type=int,  default=4,    help='number of layers for transformer architecture')
    parser.add_argument('--batch_size',      type=int,   default=256)
    parser.add_argument('--dropout',         type=float, default=0.05)
    parser.add_argument('--d_model',         type=int,   default=512)
    parser.add_argument('--d_ff',            type=int,   default=2048)
    parser.add_argument('--clip',            type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--optim',           type=str,   default='Adam')
    parser.add_argument('--lr_div',          type=float, default=2.,   help='factor to divide lr when val loss increases')
    parser.add_argument('--comments',        type=str)
    parser.add_argument('--drop_last',       type=int,   default=0)
    parser.add_argument('--pos_embeddings',  type=str,   default='absolute',      choices=['absolute', 'relative'])
    parser.add_argument('--mode',            type=str,   default='RM',            choices=['PTW', 'RM'])
    parser.add_argument('--max_seq_len',     type=int,   default=51)
    parser.add_argument('--cuda',            type=int,   default=1)
    parser.add_argument('--n_workers',       type=int,   default=0)
    parser.add_argument('--shuffle',         type=int,   default=1)
    parser.add_argument('--debug',           action='store_true')

    args = parser.parse_args()

    if args.debug:
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        args.shuffle = 0
        args.n_workers = 0
        args.log = 0 

    return args

