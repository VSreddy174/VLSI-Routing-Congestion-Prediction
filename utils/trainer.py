import torch
import os
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y, mask in tqdm(loader, desc="  Training", leave=False):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y, mask)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, mask in tqdm(loader, desc="  Validation", leave=False):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss = criterion(pred, y, mask)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_experiment(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, save_path, device):
    """
    Standardized training loop for all 6 experiments.
    Saves history and checkpoints automatically.
    """
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history
            }, os.path.join(save_path, "best_model.pth"))
            print("  ⭐ New Best Model Saved!")

        # Save Latest Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history
        }, os.path.join(save_path, "latest_checkpoint.pth"))

    return history
