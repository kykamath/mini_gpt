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