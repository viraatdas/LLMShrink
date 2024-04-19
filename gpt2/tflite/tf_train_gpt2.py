"""
TF version of GPT 2. 

This is meant to be used for TF Lite inference
"""


import os
import math
import struct
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from collections import OrderedDict
import torch




class TransformerBlock():
    pass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class Block(keras.Model):
    pass


class GPT(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = OrderedDict(
            wte = layers.Embedding(input_dim=config.vocab_size, output_dim=config.n_embd),
            wpe = layers.Embedding(input_dim=config.block_size, output_dim=config.n_embd),
            h = [Block(config) for _ in range(config.n_layer)],
            ln_f = layers.LayerNormalization(epsilon=1e-6)
        )
        
        self.lm_head = layers.Dense(config.vocab_size, use_bias=False)


        # self.transformer.wte.embeddings = self.lm_head.weight[0] # https://paperswithcode.com/method/weight-tying I might've done this wrong

    def call(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = tf.range(t, dtype=tf.int32, device=device) # shape (t,)

        # forward the GPT model itself      
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also caulate the loss
            logits = self.lm_head(x)
            loss = custom_cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import TFGPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = TFGPT2LMHeadModel.from_pretrained(model_type)

        return model_hf
    
  

def custom_cross_entropy(y_true, y_pred, ignore_index=-1):
    # Create a mask by comparing the target tensor with the ignore_index
    mask = tf.not_equal(y_true, ignore_index)
    
    # Convert mask to 1s and 0s (True becomes 1, False becomes 0)
    mask = tf.cast(mask, dtype=tf.float32)
    
    # Flatten the mask and the true labels
    y_true_flattened = tf.reshape(y_true, [-1])
    mask_flattened = tf.reshape(mask, [-1])
    
    # Calculate cross-entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_flattened, logits=y_pred)
    
    # Apply the mask to the loss
    loss *= mask_flattened
    
    # Average the loss, but only over non-ignored entries
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask_flattened)
    
    return loss

    
if __name__ == "__main__":
    import time
    import argparse
    import tiktoken

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    # if you'd like to e.g. time the forward pass only, call this script as:
    # python torch_train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    args = parser.parse_args()
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024

    # select a reasonable device to run on
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # seed the random number generators
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    write_tokenizer(enc, "gpt2_tokenizer.bin")

    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # load the GPT-2 model weights
    model = GPT.from_pretrained("gpt2")
    model.train()
    model.to(device)
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)

    # load the tokens
    # prefer to use tiny_shakespeare if it's available, otherwise use tiny_stories
    # we're using val instead of train split just because it is smaller/faster
    shake_tokens_bin = "data/tiny_shakespeare_val.bin"
    story_tokens_bin = "data/TinyStories_val.bin"
    assert os.path.isfile(shake_tokens_bin) or os.path.isfile(story_tokens_bin), "you must run prepro on some dataset"
    tokens_bin = shake_tokens_bin if os.path.isfile(shake_tokens_bin) else story_tokens_bin
    assert os.path.isfile(tokens_bin)
    print(f"loading cached tokens in {tokens_bin}")
    with open(tokens_bin, "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.int32)

    # np -> tensor, long, on device
    tokens = torch.tensor(tokens)
    tokens = tokens.to(torch.long)
    tokens = tokens.to(device)

    # lightweight dataloader
    def get_batch():
        assert B*T+1 <= len(tokens), "not enough tokens"
        # for 338,025 tokens. E.g. with B=8 T=1024, this will yield 41 batches before looping
        i = 0
        while True:
            x = tokens[i:i+B*T].view(B, T)
            y = tokens[i+1:i+B*T+1].view(B, T)
            yield x, y
            i += B*T
            if i + B*T + 1 >= len(tokens):
                i = 0 # in prod we'd want to randomize the start point a bit

    # forward backward for a few iterations
    data_iter = iter(get_batch())
    x, y = next(data_iter) # we'll overfit this batch below
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    timings = []
    for i in range(args.num_iterations):
        t0 = time.time()
        logits, loss = model(x, y)
        if not args.inference_only:
            optimizer.zero_grad()
            loss.backward()
            # on the first iteration only, save the state dict to file for later reference
            if i == 0 and args.write_tensors:
                write_model(model, "gpt2_124M.bin")
                write_state(model, x, y, logits, loss, "gpt2_124M_debug_state.bin")
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        if i > args.num_iterations - 20:
            timings.append(t1-t0)
        print(f"iteration {i}, loss: {loss.item()}, time: {(t1-t0)*1000:.3f}ms")
    if len(timings) > 0:
        print(f"final 20 iters avg: {np.mean(timings)*1000:.3f}ms")

    # before we end, let's also do one round of inference
    # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
    start = "<|endoftext|>"
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation for 16 time steps (tokens)
    max_new_tokens = 16
    temperature = 1.0
    top_k = 40
    model.eval()
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
    print('---------------')


