# this is the same as the simple version but this one performs much better because it has the Gpt tokenizer(tiktoken)

import torch
import math
import tiktoken
import os
import pickle
import sys
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 1. SETUP DEVICE ---
# This ensures it works on Nvidia GPU, Mac (MPS), or CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# --- 2. LOAD DATA ---
if not os.path.exists("input.txt"):
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- 3. TOKENIZATION (TIKTOKEN) ---
enc = tiktoken.get_encoding("gpt2")
print(f"Original Text Length (chars): {len(text)}")
encoded_text = enc.encode(text)
print(f"Tokenized Length (tokens): {len(encoded_text)}")
vocab_size = enc.n_vocab

data = torch.tensor(encoded_text, dtype=torch.long)

# Split Data
n = int(len(data) * 0.9)
train_data = data[:n]
test_data = data[n:]

block_size = 256
batch_size = 32

# --- 4. BATCH FUNCTION (FIXED) ---
def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # !!! CRITICAL FIX !!!
    # You must move the data to the same device as the model
    x, y = x.to(device), y.to(device)

    return x, y

# --- 5. MODEL CONFIG ---
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=256,
    n_embd=384,
    n_layer=6,
    n_head=6,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    use_cache=False
)

model = GPT2LMHeadModel(config)
model = model.to(device)

# --- 6. OPTIMIZER & SCHEDULER ---
steps = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-1)
scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

print("Starting training...")
model.train()

# --- 7. TRAINING LOOP ---
for step in range(steps):
    xb, yb = get_batch("train")

    outputs = model(xb, labels=yb)
    loss = outputs.loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Optional: Clip gradients to make training safer
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    if step % 500 == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"Step: {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

print("Finished training.")
