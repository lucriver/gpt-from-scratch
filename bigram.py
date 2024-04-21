import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# enable gpu
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# try to use GPU
if torch.cuda.is_available():
  device = torch.device(0)
else:
  device = torch.cpu()
print(f"Device: {torch.cuda.get_device_name(device)}")


class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
  def forward(self, idx, targets=None):
    # idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self.forward(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx,idx_next), dim=1)
    return idx
  

def get_batch(split):
  data = train_data if split == 'train' else val_data
  indexes = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in indexes])
  y = torch.stack([data[i+1:i+1+block_size] for i in indexes])
  return x, y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      _, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

input_fname = 'input.txt'
block_size = 16
batch_size = 128
max_iters = 15000
eval_iter = 1000
lr = 0.001
eval_iters = 200
val_split = 0.1

# get unique set of characters from input text
with open(input_fname,'r', encoding='utf-8') as f:
  text = f.read()
  
# get characters and vocab size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoding and decoding mapping
stoi = { char: i for i, char in enumerate(chars) }
itos = { i: char for i, char in enumerate(chars) }

# encoding and decoding functions
encode = lambda s: [stoi[char] for char in s]
decode = lambda l: "".join(itos[char] for char in l)

# get encoded version of input data 
data = torch.tensor(encode(text), dtype=torch.long).to(device)

# 90-10 train-test split of data
n = int(len(data) * val_split)
train_data = data[:n]
val_data = data[n:]

# define model
model = BigramLanguageModel(vocab_size).to(device)

# write ntp to output file before training
with open('output.txt', 'w', encoding='utf-8') as out:
  out.write('\n--------------\n')
  out.write(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=100)[0].cpu().tolist()))
  out.write('\n--------------\n')

# Define the optimizer for training
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# train
for iter in range(max_iters):
  
  # evaluation epoch
  if iter % eval_iter == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
  # update weights
  xb, yb = get_batch('train')
  logits, loss = model.forward(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
    
print(f"Loss: {loss.item()}")

# write ntp to output file before training
with open('output.txt', 'a', encoding='utf-8') as out:
  out.write(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=100)[0].cpu().tolist()))
  




