import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/news')
    parser.add_argument('--mask_pad_tokens', type=int, default=1)
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--masking', type=str, default='left_to_right')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--comments', type=str)
    return parser.parse_args()

