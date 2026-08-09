import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer  # load our subword tokenizer

parser = argparse.ArgumentParser(description='GPT Training Script')
parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training')
parser.add_argument('-epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('-save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
args = parser.parse_args()

# Determine the device to use
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = args.batch_size
epochs = args.epochs
block_size = 32
max_iters = 3000  # Total iterations for training
learning_rate = 1e-3
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

os.makedirs(args.save_dir, exist_ok=True)
writer = SummaryWriter()  # TensorBoard logger

# ---------------------------
# Load subword tokenizer
# ---------------------------
tokenizer = Tokenizer.from_file("/Users/padraigobrien/Downloads/API_LLM/.venv/bpe_tokenizer.json")
encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)
vocab_size = tokenizer.get_vocab_size()
print(f"Subword vocab size: {vocab_size}")

# ---------------------------
# Memory-map file to efficiently sample chunks from large text files
# ---------------------------
def get_random_chunk(split):
    filename = (
        "/Users/padraigobrien/Downloads/API_LLM/.venv/archive/train_split.txt"
        if split == 'train'
        else "/Users/padraigobrien/Downloads/API_LLM/.venv/archive/val_split.txt"
    )
    # Optionally, use a multiplier so you read more raw text 
    # (since subword tokenization can yield fewer tokens than raw characters)
    chunk_multiplier = 3

    while True:
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                raw_chunk_length = block_size * batch_size * chunk_multiplier
                start_pos = random.randint(0, max(file_size - raw_chunk_length, 1))
                mm.seek(start_pos)
                block = mm.read(raw_chunk_length - 1)
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
                # Encode using your subword tokenizer
                data = torch.tensor(encode(decoded_block), dtype=torch.long)
        # If the tokenized chunk is at least as long as block_size, break the loop
        if len(data) >= block_size:
            break
        else:
            pass
            #print(f"Length of Data Chunk: {len(data)} is less than block size {block_size}. Retrying...")
    
    #print(f"Length of Data Chunk: {len(data)}, Block Size: {block_size}, Batch Size: {batch_size}")
    return data


def get_batch(split):
    data = get_random_chunk(split)
    #print(f"Length of Data Chunk: {len(data)}, Length of Block Size: {block_size}, Length of Batch: {batch_size}")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        loss_total = 0
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            loss_total += loss.item()
        losses[split] = loss_total / eval_iters
    model.train()
    return losses

# ---------------------------
# Model definitions (unchanged)
# ---------------------------
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """Simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# ---------------------------
# Training setup
# ---------------------------
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Using ReduceLROnPlateau scheduler: when validation loss plateaus, reduce the LR by a factor of 0.5.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
global_step = 0

for iter in range(max_iters):
    # Periodically evaluate and log loss
    if iter % eval_iters == 0:
        losses = estimate_loss(model)
        print(f"Step {iter}: Train Loss {losses['train']:.3f}, Val Loss {losses['val']:.3f}")
        writer.add_scalar("Loss/Train", losses['train'], global_step)
        writer.add_scalar("Loss/Val", losses['val'], global_step)
        # Step the scheduler using the validation loss
        scheduler.step(losses['val'])
    
    # Get a training batch and update the model
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    global_step += 1

# Save the final model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'global_step': global_step,
}
torch.save(checkpoint, os.path.join(args.save_dir, "final_checkpoint.pt"))
writer.close()
print("Training complete, model saved.")

