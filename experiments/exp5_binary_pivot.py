import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.models import UnetDecoupled
from core.dataset import get_dataloader
from core.loss import MaskedWeightedBCE
from utils.trainer import run_experiment

# Configuration
CONFIG = {
    "data_dir": "/path/to/your/shards",
    "save_path": "./checkpoints/exp5_binary_pivot",
    "batch_size": 16,
    "lr": 5e-4,
    "epochs": 15,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 1. Load Data with is_binary=True (Thresholds targets at 0.1)
train_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='train', is_binary=True)
val_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='val', is_binary=True)

# 2. Initialize Model & Weighted BCE (h=5, v=2 to handle imbalanced hotspots)
model = UnetDecoupled(in_ch=4).to(CONFIG["device"])
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
