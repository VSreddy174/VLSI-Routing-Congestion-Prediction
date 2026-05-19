import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.models import UnetAttention
from core.dataset import get_dataloader
from core.loss import MaskedHuberLoss
from utils.trainer import run_experiment

# Configuration
CONFIG = {
    "data_dir": "/path/to/your/shards",
    "save_path": "./checkpoints/exp4_attention",
    "batch_size": 16,
    "lr": 8e-4,
    "epochs": 15,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 1. Load Data
train_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='train')
val_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='val')

# 2. Initialize Attention Model & Huber Loss
model = UnetAttention(in_ch=4).to(CONFIG["device"])
criterion = MaskedHuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 3. Run
if __name__ == "__main__":
    run_experiment(
        model, train_loader, val_loader, optimizer, 
        criterion, scheduler, CONFIG["epochs"], 
        CONFIG["save_path"], CONFIG["device"]
    )
