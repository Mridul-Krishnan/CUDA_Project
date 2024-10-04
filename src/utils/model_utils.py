import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, loss, model_name, checkpoint_dir="checkpoints"):
    """Saves the model checkpoint separately for each model."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    filename = f"{model_name}_checkpoint_epoch_{epoch}.pth.tar"
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"Checkpoint for {model_name} saved at {os.path.join(checkpoint_dir, filename)}")


def load_checkpoint(model, optimizer, scheduler=None, checkpoint_path="checkpoints", model_name=None):
    """Loads the model checkpoint and returns the epoch and loss."""
    if not model_name:
        raise ValueError("model_name must be specified.")
    
    checkpoint_file = os.path.join(checkpoint_path, f"{model_name}_checkpoint_epoch_latest.pth.tar")
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint for {model_name} loaded. Resuming from epoch {epoch} with loss {loss:.4f}.")
    
    return epoch, loss

def load_checkpoint2(model, optimizer, scheduler=None, checkpoint_path="checkpoints"):
    """Loads the model checkpoint and returns the epoch and loss."""
    
    checkpoint_file = checkpoint_path
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model_name = checkpoint_file.split("/")[-1]
    print(f"Checkpoint for {model_name} loaded. Resuming from epoch {epoch} with loss {loss:.4f}.")
    
    return model, optimizer, scheduler, epoch, loss