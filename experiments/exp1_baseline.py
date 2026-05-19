import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.models import UnetBaseline
from core.dataset import get_dataloader
from core.loss import MaskedHuberLoss # Using Huber with low delta acts like MSE
from utils.trainer import run_experiment

# Configuration
CONFIG = {
    "data_dir": "/path/to/your/shards",
    "save_path": "./checkpoints/exp1_baseline",
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 15,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 1. Load Data
train_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='train')
val_loader = get_dataloader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], split='val')

# 2. Initialize Model, Loss, Optimizer
model = UnetBaseline(in_ch=4).to(CONFIG["device"])
criterion = torch.nn.MSELoss() # Direct MSE for the baseline
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 3. Run Training
if __name__ == "__main__":
    run_experiment(
        model, train_loader, val_loader, optimizer, 
        criterion, scheduler, CONFIG["epochs"], 
        CONFIG["save_path"], CONFIG["device"]
    )
