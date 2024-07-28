import torch
import torch.nn as nn
from torch.nn import functional as F
from model_add import TransformerModel
from datetime import datetime

# hyperparameters
batch_size = 64 # number of independent sequences processed in parallel
block_size =  64 # context size
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embed = 64
n_head = 4
n_layer = 4
# -----------------

torch.manual_seed(1337)

chars = ['0','1','2','3','4','5','6','7','8','9','$','+','=','.', ' ', '[',']',',','h','a','s','n','u','m','b','e','r','s'] # All the chars needed for addition
vocab_size = len(chars)
# tokenize chars
stoi = {x:i for i,x in enumerate(chars)}
itos = {i:x for i,x in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join([itos[x] for x in xs])

# Different formats for the training data

def getPlainFormat(data):
    return"".join([f'{a}+{b}={str(c)}' for [a, b], c in zip(data.tolist(), data.sum(1).tolist())])


def getReverseFormat(data):
    return"".join([f'${a}+{b}={str(c)[::-1]}$' for [a, b], c in zip(data.tolist(), data.sum(1).tolist())])

# Data needs 3 numbers
def getChainOfThoughtData(data):
    return"".join([f'[{a}, {b}, {c}] has 3 numbers. {a} + {b} = {str(a+b)[::-1]}. {a+b} + {c} = {str(a+b+c)[::-1]}. {a} + {b} + {c} = {str(a+b+c)[::-1]}$' for [a, b, c] in data.tolist()])

#----------------------------------------

# Generate batches on the fly
def getRandomData(n=1000):
    data = torch.cat([torch.randint(10, (n, 2))])
    return torch.tensor(encode(getReverseFormat(data)), dtype=torch.long)

def getBatch():
    data = getRandomData()
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
            x, y = getBatch()
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean() # Average the losses to make loss less noisy
    model.train()
    return out

def printSampleFromModel(context, max_tokens):
   print(decode(model.generate(idx=context, max_tokens=max_tokens)[0].tolist()))

m = TransformerModel(vocab_size=vocab_size, block_size=block_size, n_embed=n_embed, n_head=n_head, n_layer=n_layer)
model = m.to(device)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for steps in range(max_iters):

    if(steps % eval_interval == 0):
        losses = estimate_loss()
        print(f'step {steps}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')
    
    # sample data
    xb, yb = getBatch()

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

now = datetime.today().strftime('%Y-%m-%d-%H-%M')
torch.save(model, f'./models/model_add2_{now}.pth')
print(f'./models/model_add2_{now}.pth')
# context = torch.zeros((1,1), dtype=torch.long, device=device)
context = torch.tensor(encode('2+5='), dtype=torch.long).view(1,-1)
print(context)
printSampleFromModel(context, 2)