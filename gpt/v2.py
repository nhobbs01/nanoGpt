import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8 # context size
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
n_heads = 4
# -----------------

torch.manual_seed(1337)

#https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# read it in to inspect it
with open('../input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# tokenize chars
stoi = {x:i for i,x in enumerate(chars)}
itos = {i:x for i,x in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join([itos[x] for x in xs])


# Data splits
data = torch.tensor(encode(text), dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

def getBatch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size, (batch_size,)) ## len(data) - block_size so we don't index out of range
    x = torch.stack([data[i:block_size+i] for i in ix])
    y = torch.stack([data[i+1:block_size+i+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y 

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = getBatch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean() # Average the losses to make loss less noisy
    model.train()
    return out

def printSampleFromModel(context, max_tokens):
   print(decode(model.generate(idx=context, max_tokens=max_tokens)[0].tolist()))

# ---------------- MODELS

class Head(nn.Module):
    """ Single head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, head_size (B, T, 16)
        q = self.query(x) # B, T, head_size (B, T, 16)

        wei = q @ k.transpose(-2,-1) # (B, T, 16) @ (B, 16, T) => (B, T, T)
        wei *= C**-0.5 # Scale attention
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf')) # Doesn't communitcate with the past
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHead(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads =nn.ModuleList([Head(head_size) for i in range(n_heads)])

    def forward(self, x):
        outs = [head(x) for head in self.heads]
        out = torch.cat(outs, -1)
        return out


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHead(n_heads, n_embed//n_heads)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # B, T, C  (C is n_embed)
        position_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C (create an embedding for each time step)
        x = token_emb + position_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is None:
            loss = None
        else: 
            # Need to reshape for cross_entropy
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # 32 65
            targets = targets.view(B*T) # 32
            loss = F.cross_entropy(logits, targets)
        return logits, loss        

    def generate(self, idx, max_tokens):
        
        # idx is (B, T) array of current context
        for _ in range(max_tokens):
            # get the predictions
            logits, loss= self(idx[:,-block_size:])
            # Focus on the last time dimension
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = TransformerModel()
model = m.to(device)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for steps in range(max_iters):

    if(steps % eval_interval == 0):
        losses = estimate_loss()
        print(f'step {steps}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')
    
    # sample data
    xb, yb = getBatch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
printSampleFromModel(context, 500)


"""
LOG:

with one self-attention head
- val loss about 2.4

simple multi head attention - 8 heads
 - val loss 2.21

simple multi head attention - 4 heads
 - val loss 2.27


"""