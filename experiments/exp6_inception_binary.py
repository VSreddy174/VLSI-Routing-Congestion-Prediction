import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.models import InceptionUNet
from core.dataset import get_dataloader
from core.loss import MaskedWeightedBCE
from utils.trainer import run_experiment

# Configuration
CONFIG = {
    "data_dir": "/path/to/your/shards",
    "save_path": "./checkpoints/exp6_inception_binary",
    "batch_size": 16,
    "lr": 2e-4, # Lower LR for finer convergence in complex Inception layers
    "epochs": 15,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 1. Load Data (Binary)
train_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='train', is_binary=True)
val_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='val', is_binary=True)

# 2. Initialize Inception Model & Weighted BCE
model = InceptionUNet(in_ch=4).to(CONFIG["device"])
criterion = MaskedWeightedBCE(h_weight=5.0, v_weight=2.0)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 3. Run
if __name__ == "__main__":
    run_experiment(
        model, train_loader, val_loader, optimizer, 
        criterion, scheduler, CONFIG["epochs"], 
        CONFIG["save_path"], CONFIG["device"]
    )
