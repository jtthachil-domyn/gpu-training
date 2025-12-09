import os
import json
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# -----------------------------------------------------------------------------
# Logging Setup (Rank Aware)
# -----------------------------------------------------------------------------
def setup_logging(rank, save_dir):
    # Ensure log directory exists
    log_dir = os.path.join(save_dir, f"rank_{rank}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to file and console
    logging.basicConfig(
        format=f"%(asctime)s - [Rank {rank}] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Tiny Model Definition (Redefined for standalone clarity)
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
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
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

class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(config["hidden_size"]),
                "attn": TinyAttention(config["hidden_size"], config["num_attention_heads"]),
                "ln2": nn.LayerNorm(config["hidden_size"]),
                "mlp": TinyMLP(config["hidden_size"], config["intermediate_size"])
            }) for _ in range(config["num_hidden_layers"])
        ])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, input_ids, labels=None):
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = x + block["attn"](block["ln1"](x))
            x = x + block["mlp"](block["ln2"](x))
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------
# Fake Dataset
# -----------------------------------------------------------------------------
class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        data = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": data, "labels": data}

# -----------------------------------------------------------------------------
# DDP Logic
# -----------------------------------------------------------------------------
def ddp_main():
    # Parse args (no arguments needed for basic torchrun, but nice to have)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default="config/config.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # DDP Environment Variables set by torchrun
    # LOCAL_RANK: Rank on the current node
    # RANK: Global rank
    # WORLD_SIZE: Total number of processes
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Init Process Group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    # Setup Logging
    logger = setup_logging(rank, save_dir="logs")
    logger.info(f"Initialized DDP Process: Global Rank {rank}, Local Rank {local_rank}, World Size {world_size}")

    # Load Config
    with open(args.model_config, 'r') as f:
        config = json.load(f)

    # Device
    # For simulation on Mac, we use CPU or MPS.
    # DDP on MPS is experimental, usually we simulate with CPU for 'gloo' backend if MPS has issues.
    # Docs say "LOCAL execution uses CPU/MPS". Gloo works on CPU.
    # We will just use CPU to be safe for Gloo distributed simulation on Mac unless user explicitly configured MPS for DDP (which is rare/complex).
    # Actually, let's try to map to device if possible, but CPU is safest for pure Gloo simulation without CUDA.
    device = torch.device("cpu") 
    logger.info(f"Running on device: {device}")

    # Data
    dataset = SyntheticTextDataset(config["vocab_size"], 64, 1000)
    
    # DISTRIBUTED SAMPLER
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Model
    model = TinyTransformer(config).to(device)
    
    # In DDP, we wrap the model. 
    # Since we are on CPU/Gloo, we use DistributedDataParallel.
    # Note: On CPU, device_ids should be None.
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    logger.info("Starting training...")
    
    # Training Loop
    for epoch in range(args.epochs):
        # IMPORTANT: Set epoch for sampler shuffling
        sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=labels)
            
            loss.backward()
            
            # Gradients are synchronized here automatically by DDP
            optimizer.step()
            
            if step % 10 == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # Dist Barrier
        dist.barrier()
        
        # Checkpointing (Rank 0 only)
        if rank == 0:
            ckpt_path = f"checkpoints/ddp_epoch_{epoch}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.module.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            
    dist.destroy_process_group()
    logger.info("Process group destroyed. Exiting.")

if __name__ == "__main__":
    # Standard boilerplate to start DDP
    ddp_main()