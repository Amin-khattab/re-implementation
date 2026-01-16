import torch
import torch.nn as nn
from torch.nn import functional as F

# --- FIX 1: Defined n_layer so the model doesn't crash ---
n_layer = 4

with open("input.txt","r",encoding="utf-8") as f: # reads the entire code
    text = f.read()

# creats a sorted list that contains eniuqe chars and we have a voocab size of 65
chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating the string to int by going from charictar to number by the enumerate function
stoi = {ch:i for i,ch in  enumerate(chars)}
# creating the int to string by going from charictar to number by the enumerate function
iots = {i:ch for i,ch in enumerate(chars)}

# creating the encoder and decoder
encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: "".join([iots[i]for i in l ])

# we encode the whole data idk why we did torch.long tbh
data = torch.tensor(encoder(text),dtype=torch.long)

# we split the data into train and test data
n = int(len(data) * 0.9)

train_data = data[:n]
test_data = data[n:]
# i think this part is only for understanding that the og bigram model only wanted the last character and we created a context and target for it
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]

# this is only to get the same result as video
torch.manual_seed(1337)

batch_size = 4
block_size = 8

# this randomly looks at the data and takes 8 chars (block size) somewhere random we take 4 rows of 8 chars and we stack them so the look like a matrix
def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size , (batch_size,))
    x =torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')

# this class refers to a head

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        # here we create key,query,value and the size is 32 and bias is False so that nothing interfers with our k,q,v values
        self.key = nn.Linear(32,head_size,bias=False)
        self.query = nn.Linear(32,head_size,bias=False)
        self.value = nn.Linear(32,head_size, bias = False)
        # we dont have a tril in pytorch thats why we do a reggister buffer then we create a tril of 8 by 8
        self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size)))
        #dropout of %20 for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        # we get the shape of x
        B,T,C = x.shape
        # we create the k,q,v
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # we multiply quesry by k so that we know the highest affinity is wich one but idk why we transpose
        # and for the * (k.shape[-1] ** -0.5) we do that because its a part of the function but later we softmax so we dont want a bold softmax we multiply it by a smaller number
        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)

        # we mask it so that the future tokens are hidden
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # we softmax them so that the numbers are between 1-0
        wei = F.softmax(wei,dim = -1)
        #drop out again
        wei = self.dropout(wei)
        # then we multimply the wei with the values and we now know the highest affinity between the tokens
        out = wei @ v

        return out

# this class represents the feedforward layer in attention
# it adds a bit of thinking like when prediction happens for h that e comes after it thinks a bit like this is either he or hello
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # and why did we multiply by 4
            nn.Linear(n_embd,n_embd * 4),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2)

        )
    def forward(self,x):
        return self.net(x)

# this class is for a bunch of heads at the same time
class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        #this calls the Head class and gives it a size of the argument for the number of heads that we want
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #idk what this projection does tbh
        self.proj = nn.Linear(32,32)
        #dropout again
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        #Head 1 looks for grammar.

        #Head 2 looks for rhymes.

        #Head 3 looks for names.

        # Concatenation: You take their reports and staple them together side-by-side.
        # Projection (self.proj): This is a "Manager" layer. It reads the stapled reports and
        # summarizes them into a single, unified vector that the rest of the network can understand.

        out =  torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

# this block class does all the few classes we just created as many times as we want
class Block(nn.Module):
    def __init__(self,n_embd,num_head):
        super().__init__()
        # i think we will get 8 as the head_size
        head_size = n_embd // num_head
        # calling the multihead
        self.sa_head = MultiHeadAttention(num_head,head_size)
        #calling the feedforward
        self.feed = FeedForward(n_embd)
        #dong layer norm a form of regularization that normalizes data as rows but idk why we cretaed 2 layers norms because one is for the attention and the other is for feedforward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        # here we do the residual connection and layer norm as we said
        x = x +  self.sa_head(self.ln1(x))
        x = x + self.feed(self.ln2(x))
        return x

# this is the model that does everything

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        # we create an emedding table for all the chars and for each one there are a 32 nums that represent it
        self.token_embeding_table = nn.Embedding(vocab_size, 32)
        # we create the positional embedding table so that we know where they came from but dont know exactly how
        self.positional_embedding = nn.Embedding(block_size, 32)
        # --- FIX 2: Fixed the syntax for creating the blocks list ---
        # this creates a sequential of the class block that gets called as many times as n_layer
        self.block = nn.Sequential(*[Block(32,4) for _ in range(n_layer)])
        #a layer norm
        self.norm = nn.LayerNorm(32)
        # and we have a lm_head idk why tbh
        self.lm_head = nn.Linear(32, vocab_size)

    def forward(self,idx,targets = None):
        B,T = idx.shape
        # we tokenize
        tk_embd = self.token_embeding_table(idx)
        #we take the position
        ps_embed = self.positional_embedding(torch.arange(T,device=idx.device))
        # we add them together
        x = tk_embd + ps_embed
        # we take that x and apply the Multihead attenation as many times as we want
        x = self.block(x)
        # norm it
        x = self.norm(x)
        # and then we finally get the logits which are the number that x carries but when softmaxed it adds up to 1 and can be predicted
        logits = self.lm_head(x)
        # we get the loss if there was one
        if targets is None:
            loss = None
        else:
            # B = Batch is for how many parralel batchs are there
            # T = is for the length of the batch
            # C = is how many vectors represent this char in the embedding table
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits,targets)

        return logits,loss
    # we generate by this function
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # this limits the batch size to not be higher than 8
            idx_cond = idx[:, -self.block_size:]
            # we get the logits/loss here
            logits, loss = self(idx_cond)
            # i think here we only take the T dim for the position of it
            logits = logits[:, -1, :]
            # we do a softmax idk why
            probs = F.softmax(logits, dim=-1)
            # it predicts the next token in line
            idx_next = torch.multinomial(probs, num_samples=1)
            # then we concactnate and use it to predict the other one
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Running the training loop briefly so we get to the end ---
m = BigramLanguageModel(vocab_size,block_size)
out,loss = m(xb,yb)
optimizer = torch.optim.AdamW(m.parameters(),lr= 1e-3)

# the steps it trains
for steps in range(10):
    xb, yb = get_batch('train')
    logits , loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training Loss:", loss.item())

# --- THE PART YOU WANT TO SEE ---

B,T,C = 4,8,32
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C,head_size,bias=False)
query =nn.Linear(C,head_size,bias=False)
value =nn.Linear(C,head_size,bias=False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2,-1) * head_size ** -0.5

tril = torch.tril(torch.ones(T,T))

print("\n--- MATRIX: TRIL (The Mask) ---")
print(tril) # FIX 3: Changed 'print(tr)' to 'print(tril)'

wei = wei.masked_fill(tril == 0 , float("-inf"))

print("\n--- MATRIX: WEI (The Affinities) ---")
print(wei[0]) # Printing just the first batch so it is readable

wei = F.softmax(wei,dim=-1)
print(wei[0])
v = value(x)
xbow3 =wei @ v
