### transformer-MADE

Expected perplexities when running with default params, both approaches give around 115 valid ppl

```
python main.py --model transformer --data_dir data/ptb --masking left_to_right 
python main.py --model transformer --data_dir data/ptb --masking random        
```

Optimizing it like an LSTM
```
python main.py --model transformer --data_dir data/ptb --optim SGD --lr 20 --batch_size 128 --n_layers 8 --n_heads 16 --masking left_to_right    # 142 valid ppl
python main.py --model transformer --data_dir data/ptb --optim SGD --lr 20 --batch_size 128 --n_layers 8 --n_heads 16 --masking random           # 110 valid ppl
```
