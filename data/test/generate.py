import sys
import os
import numpy as np

vocab_size = 20
seq_len = 10
out_file = open(sys.argv[1], 'w')

for i in range(10000):
    xx = np.random.randint(vocab_size)
    line = [(xx + i) % vocab_size for i in range(seq_len)]
    string = ''

    for piece in line: 
        string += ' ' + str(piece)

    out_file.write(string + '\n')


