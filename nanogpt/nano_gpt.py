# https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch


text = None
with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()


# model produces only these chars and vocab_size
chars = sorted(list(set(text)))
vocab_size = len(chars) 

# tokenize --> character level encoding decoding

## char to index
stoi = {ch: i for i, ch in enumerate(chars)}

## index to char
itos = {i: ch for i, ch in enumerate(chars)}  

encode = lambda x: [stoi[ch] for ch in x]
decode = lambda l: ''.join([itos[i] for i in l])

'''
tokenizer:
Google uses sentencepiece
'''


data = torch.tensor(encode(text), dtype=torch.long)

## Train/test -> 90% train, 10% test
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# context_size is synonmyous with block_size
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target= y[t]
    print(f"Context: {context} -> Target: {itos[target]}")
  
torch.manual_seed(1337)
batch_size = 4 # how many independent seuqneces will we process in parallel?
block_size = 16 # what is the maximum context length for prediciotns?

def get_batch(split):
   pass
   