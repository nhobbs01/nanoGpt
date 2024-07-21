import torch
import torch.nn as nn
from torch.nn import functional as F
from model import TransformerModel

# hyperparameters
batch_size = 32 # number of independent sequences processed in parallel
block_size = 8 # context size
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
n_head = 4
n_layer = 4
# -----------------


nn.Sequential()
torch.manual_seed(1337)

#https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# read it in to inspect it
with open('./input.txt', 'r', encoding='utf-8') as f:
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

m = TransformerModel(vocab_size=vocab_size)
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

torch.save(model, 'model_v2.pth')
context = torch.zeros((1,1), dtype=torch.long, device=device)
printSampleFromModel(context, 500)


"""
LOG:
all with these params
batch_size = 32
block_size = 8 # context size
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
n_head = 4

with one self-attention head
- val loss ~2.4

simple multi head attention - 4 heads
 - val loss ~2.27

with feed forward layer after multi head attention 
 - val loss ~2.23

with layer normalization + residual connections (1 block)
 - val loss ~2.21

with 4 blocks (multi-head self attention, norm + add, feed forward)
 - val loss  ~2.22

with applying the norm before the transformation (this is a deviation from the paper but how it is done now)
- val loss ~2.14

"""