#!/usr/bin/env python3
"""
Simple example of loading a checkpoint and setting it to training mode
"""
import torch
from pathlib import Path

def load_checkpoint_for_training_simple(checkpoint_path):
    """
    Simple function to load checkpoint and set to training mode
    
    Args:
        checkpoint_path: Path to your checkpoint file
    
    Returns:
        dict with model, optimizer_state, epoch info
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📂 Loaded checkpoint: {checkpoint_path}")
    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get the model state dict
    model_state = checkpoint['model_state_dict']
    
    # Remove DataParallel wrapper if present
    if list(model_state.keys())[0].startswith('module.'):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    # Extract training info
    epoch = checkpoint.get('epoch', 0)
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    scheduler_state = checkpoint.get('scheduler_state_dict', None)
    
    print(f"📈 Last epoch: {epoch}")
    print(f"🔧 Has optimizer state: {optimizer_state is not None}")
    print(f"📅 Has scheduler state: {scheduler_state is not None}")
    
    return {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'epoch': epoch,
        'checkpoint': checkpoint
    }


def set_model_to_training_mode(model):
    """
    Set model to training mode and enable gradients
    
    Args:
        model: PyTorch model
    """
    
    # Set to training mode
    model.train()
    
    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    print("🎯 Model set to training mode")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def example_usage():
    """Example of how to use checkpoint for training"""
    
    # Available checkpoints (choose one that exists)
    possible_checkpoints = [
        "kaggle_asr_outputs/checkpoints/best_model.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_50.pt", 
        "checkpoints/best_model_scripts1_fixed.pt",
        "best_model.pt"
    ]
    
    # Find first existing checkpoint
    checkpoint_path = None
    for cp in possible_checkpoints:
        if Path(cp).exists():
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoints found!")
        return
    
    print(f"🚀 Using checkpoint: {checkpoint_path}")
    
    # Step 1: Load checkpoint
    checkpoint_info = load_checkpoint_for_training_simple(checkpoint_path)
    
    # Step 2: Create your model (you need to import your actual model class)
    # For example:
    # from models.konkanivani_asr import KonkaniVaniASR
    # model = KonkaniVaniASR(vocab_size=..., ...)
    # model.load_state_dict(checkpoint_info['model_state_dict'])
    
    print("\n✅ Checkpoint loaded successfully!")
    print("📝 Next steps:")
    print("1. Create your model instance")
    print("2. Load the model_state_dict into your model")
    print("3. Set model to training mode with model.train()")
    print("4. Create optimizer and load optimizer_state_dict if available")
    print("5. Start training from epoch + 1")
    
    # Example of what the training setup would look like:
    print("\n📋 Example training setup:")
    print("""
    # Load your model
    model = YourModelClass(...)
    model.load_state_dict(checkpoint_info['model_state_dict'])
    model.train()  # Set to training mode
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if checkpoint_info['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
    
    # Resume from next epoch
    start_epoch = checkpoint_info['epoch'] + 1
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Your training code here
        pass
    """)


if __name__ == "__main__":
    example_usage()