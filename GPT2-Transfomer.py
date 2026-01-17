import torch
import math
import tiktoken
from transformers import GPT2Config,GPT2LMHeadModel
from torch.optim.lr_scheduler import CosineAnnealingLR

with open("input.txt","r",encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
print(f"Original Text Length (chars): {len(text)}")
encoded_text = enc.encode(text)
print(f"Tokenized Length (tokens): {len(encoded_text)}")
print("--- Notice how the tokenized length is shorter? Compressed info! ---")
vocab_size = enc.n_vocab

data = torch.tensor(encoded_text,dtype=torch.long)

n = int(len(data) * 0.9)

train_data = data[:n]
test_data = data[n:]
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]

block_size = 256
batch_size = 64

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size , (batch_size,))
    x =torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

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
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

steps = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=1e-1)
scheduler = CosineAnnealingLR(optimizer, T_max=steps,eta_min=1e-5 )


model.train()

for step in range(steps):
    xb,yb = get_batch("train")

    outputs = model(xb,labels = yb)

    loss = outputs.loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 500 == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"Step: {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
print("finished training ")
