"""
TF version of GPT 2. 

This is meant to be used for TF Lite inference
"""


import argparse
import os
import math
import struct
from dataclasses import dataclass
import time

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from collections import OrderedDict
import torch

class NewGELU(layers.Layer):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def call(self, input):
        return 0.5 * input * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * tf.pow(input, 3.0))))

class CausalSelfAttention(layers.Layer):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = layers.Dense(3 * config.n_embd)
        self.c_proj = layers.Dense(config.n_embd)
        self.config = config

    def build(self, input_shape):
        self.bias = tf.linalg.band_part(tf.ones((self.config.block_size, self.config.block_size)), -1, 0)

    def call(self, x):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], self.n_embd
        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, 3, axis=2)
        q = self.split_heads(q, B)
        k = self.split_heads(k, B)
        v = self.split_heads(v, B)

        # Attention with causal mask
        weights = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.n_embd)
        mask = self.bias[:T, :T]
        weights += (mask - 1) * 1e9  # Apply mask
        attention_output = tf.matmul(tf.nn.softmax(weights), v)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (B, T, C))
        return self.c_proj(attention_output)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.n_embd // self.n_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    

class MLP(layers.Layer):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = layers.Dense(4 * config.n_embd)
        self.c_proj = layers.Dense(config.n_embd)
        self.gelu = NewGELU()

    def call(self, x):
        x = self.gelu(self.c_fc(x))
        return self.c_proj(x)


class Block(layers.Layer):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(config)

    def call(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

    @tf.function
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens based on the input sequence, using the model to predict subsequent tokens.

        This method performs token generation iteratively, extending the sequence one token at a time.
        It controls the sharpness of token probability distribution using temperature, and optionally limits
        the sampling pool to the top_k most likely next tokens.
        """

        # Ensure model is in evaluation mode
        self.model.trainable = False

        for _ in range(max_new_tokens):
            # Forward pass through the model
            logits = self.model(input_ids)[0][:, -1, :]  # Get the logits for the last token

            # Apply temperature scaling
            logits = logits / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                values, indices = tf.nn.top_k(logits, k=top_k)
                min_values = tf.reduce_min(values, axis=-1, keepdims=True)
                logits = tf.where(logits >= min_values, logits, tf.fill(tf.shape(logits), -float('inf')))

            # Calculate softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1)

            # Sample from the probability distribution
            next_token_ids = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)

            # Append sampled index to the running sequence and continue
            input_ids = tf.concat([input_ids, next_token_ids], axis=-1)

        return input_ids
        
        
    
  

# Utility methods
def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_tensors(model_tensors, L, file):
    write_fp32(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fp32(model_tensors["transformer.wpe.weight"], file) # (T, C)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L): # (L, 3C, C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, 3C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L): # (L, C, C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L): # (L, 4C, C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L): # (L, C, 4C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fp32(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fp32(model_tensors["transformer.ln_f.bias"], file) # (C, )

def write_model(model, filename):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # magic
    header[1] = 1 # checkpoint version = 1
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters on CPU are next
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # now write
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # model parameters
        write_tensors(params, model.config.n_layer, file)
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 1 # run state version = 1
    header[2] = x.size(0) # batch size of the batch, B
    header[3] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file)
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 1 # tokenizer version = 1
    header[2] = n # number of tokens
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")



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
    # python tf_train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024
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
    assert 1 <= T <= 1024, "Sequence length must be between 1 and 1024"

    # Select a reasonable device to run on 
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            device = '/GPU:0'
        except RuntimeError as e:
            print(e)
            device = '/CPU:0'
    else:
        device = '/CPU:0'

    print(f"Using device: {device}")

    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
                                  
    write_tokenizer(enc, "gpt2_tokenizer.bin")

    if args.tensorcores:
        tf.keras.mixed_precision.set_global_policy('float_32')

    model = GPT.from_pretrained("gpt2")

    if args.compile:
        print("compiling the model...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    

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
    tokens = tf.convert_to_tensor(tokens, dtype=tf.int32)

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

    # Forward-backward for a few iterations

    data_iter = iter(get_batch())
    x, y = next(data_iter) # we'll overfit this batch below

    # Create an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Define the training loop
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits, loss = model(x, y, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    timings = []
    for i in range(args.num_iterations):
        t0 = time.time()
        loss, logits = train_step(x, y)
        
        if i == 0 and args.write_tensors:
            model.save_weights("gpt2_124M.weights")
            # Write state is not straightforward in TensorFlow

        if i > args.num_iterations - 20:
            timings.append(time.time() - t0)
        
        print(f"iteration {i}, loss: {loss}, time: {(time.time() - t0) * 1000:.3f}ms")

    if len(timings) > 0:
        print(f"final 20 iters avg: {np.mean(timings) * 1000:.3f}ms")

    # Inference
    start = "<|endoftext|>"
    start_ids = encode(start)
    x = tf.constant([start_ids], dtype=tf.int32)

    max_new_tokens = 16
    temperature = 1.0
    top_k = 40

    # TensorFlow has a built-in beam search function
    outputs = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(outputs[0].numpy().tolist()))
    print('---------------')