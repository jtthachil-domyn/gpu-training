import os
import json
import math
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Tiny Model Components (From Scratch)
# -----------------------------------------------------------------------------

class TinyAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        # (B, T, 3 * hidden_size) -> (B, T, 3, num_heads, head_dim)
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        # Permute to (3, B, num_heads, T, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Attention scores: (B, num_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # mask is (B, 1, T, T) or similar broadcastable shape
            attn = attn.masked_fill(mask == 0, float("-inf"))
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output: (B, num_heads, T, head_dim)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TinyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

class TinyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.attn = TinyAttention(
            config["hidden_size"], 
            config["num_attention_heads"], 
            config["attention_probs_dropout_prob"]
        )
        
        self.ln2 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.mlp = TinyMLP(
            config["hidden_size"], 
            config["intermediate_size"], 
            config["hidden_dropout_prob"]
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.pos_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.layer_norm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
        self.blocks = nn.ModuleList([
            TinyBlock(config) for _ in range(config["num_hidden_layers"])
        ])
        
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Weight tying
        self.head.weight = self.embeddings.weight

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        # Basic causal mask: (B, 1, T, T)
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)
        
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        x = self.embeddings(input_ids) + self.pos_embeddings(pos_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask)
            
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config["vocab_size"]), shift_labels.view(-1))
            
        return logits, loss

# -----------------------------------------------------------------------------
# Fake Dataset (Synthetic Text)
# -----------------------------------------------------------------------------
class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random integer sequence
        data = torch.randint(0, self.vocab_size, (self.seq_len,))
        # For causal LM, labels are typically same as input (shifted inside model)
        return {"input_ids": data, "labels": data}

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train(args):
    # Load config
    with open(args.model_config, 'r') as f:
        config = json.load(f)
        
    logger.info(f"Loaded config: {config}")
    
    # Setup Device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Setup Data
    dataset = SyntheticTextDataset(
        vocab_size=config["vocab_size"],
        seq_len=64,
        num_samples=1000
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup Model
    model = TinyTransformer(config).to(device)
    logger.info("Model initialized.")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training Loop
    global_step = 0
    model.train()
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=labels)
            
            loss.backward()
            optimizer.step()
            
            # Logging
            if global_step % args.log_interval == 0:
                logger.info(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                
            global_step += 1
            
        # Checkpoint at end of epoch
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        
    writer.close()
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default="config/config.json", help="Path to model config")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-dir", type=str, default="logs/local_tiny")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    
    args = parser.parse_args()
    
    train(args)