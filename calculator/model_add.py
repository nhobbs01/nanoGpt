import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout=0.2
# -----------------

# ---------------- MODELS
class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, head_size, block_size, n_embed):
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.n_head = n_head
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T,self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T,self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # q @ k.T => (B, n_head,T, head_size) @ (B, n_head, T, head_size) => (B, n_head, T, T)
        wei = (q@ k.transpose(-2, -1))
        wei *= self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.drop(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size ,n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # Normalize the last dim (C) which is n_embed
        self.ffwd = FeedForward(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.sa(self.ln1(x))+ x
        x = self.ffwd(self.ln2(x)) + x
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
        

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # B, T, C  (C is n_embed)
        position_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C (create an embedding for each time step)
        x = token_emb + position_emb # (B, T, C)
        x = self.blocks(x)
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
            logits, loss= self(idx[:,-self.block_size:])
            # Focus on the last time dimension
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if(idx_next.item() == 10):
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx