import torch.nn as nn
from torch.nn import functional as F
import math
import torch

# --- Hyperparameters and Setup ---
BATCH_SIZE = 64      # How many independent sequences we process in parallel
BLOCK_SIZE = 256     # The maximum context length (T in previous parts)
MAX_ITERS = 5000     # Total training iterations
EVAL_INTERVAL = 500  # How often to run validation
LEARNING_RATE = 3e-4
N_EMBD = 384         # The dimension of the token embeddings (d_model)
N_HEAD = 6           # Number of attention heads
N_LAYER = 6          # Number of Decoder blocks to stack
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Set seed for reproducibility
torch.manual_seed(1337)

# 1. Get the dataset (Tiny Shakespeare)
# Download the data if needed (running in a local environment)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Character-level Tokenization (Vocabulary)
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 3. Train/Validation Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- Data Loader Utilities ---
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # ix is a tensor of random starting indices for the subsequences
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # x is the input sequence (context), y is the target (next token)
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- Evaluation Utility ---
@torch.no_grad()
def estimate_loss(model):
    """ Evaluates the model on train/val split and returns average loss."""
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_INTERVAL)
        for k in range(EVAL_INTERVAL):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

############# Custom code ##############

class AddModule(nn.Module):
    def __init__(self):
        super(AddModule, self).__init__()

    def forward(self, x, y):
        return x + y




# d = get_batch("train")
# print(d)