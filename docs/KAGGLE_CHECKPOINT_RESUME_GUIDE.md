# Kaggle Checkpoint Resume Guide

## Quick Changes for Your Existing Notebook

### 1. Update Checkpoint Loading Path

**Change this:**
```python
checkpoint_path = "best_model.pt"
checkpoint = torch.load(checkpoint_path)
```

**To this:**
```python
# Kaggle input dataset path
checkpoint_path = "/kaggle/input/your-checkpoint-dataset/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
```

### 2. Handle DataParallel Wrapper

**Add this after loading checkpoint:**
```python
# Remove DataParallel wrapper if present
model_state = checkpoint['model_state_dict']
if list(model_state.keys())[0].startswith('module.'):
    model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    checkpoint['model_state_dict'] = model_state
```

### 3. Set Model to Training Mode

**Add this after loading model state:**
```python
# Load model state
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# CRITICAL: Set to training mode
model.train()

print(f"🎯 Model set to training mode: {model.training}")
```

### 4. Resume from Correct Epoch

**Add this:**
```python
start_epoch = checkpoint.get('epoch', 0) + 1
print(f"📈 Resuming from epoch: {start_epoch}")

# In your training loop:
for epoch in range(start_epoch, start_epoch + num_additional_epochs):
    # Your training code
    pass
```

### 5. Load Optimizer State (Optional but Recommended)

**Add this:**
```python
# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Load previous optimizer state if available
if 'optimizer_state_dict' in checkpoint:
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Loaded optimizer state")
    except:
        print("⚠️ Using fresh optimizer")
```

## Complete Example Cell

Here's a complete cell you can add to your notebook:

```python
def resume_from_checkpoint(checkpoint_path, model, device):
    """Resume training from checkpoint"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel wrapper
    model_state = checkpoint['model_state_dict']
    if list(model_state.keys())[0].startswith('module.'):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    # Load model state
    model.load_state_dict(model_state)
    model.to(device)
    
    # SET TO TRAINING MODE
    model.train()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    print(f"✅ Resumed from epoch {start_epoch}")
    print(f"🎯 Training mode: {model.training}")
    
    return model, optimizer, start_epoch

# Usage:
checkpoint_path = "/kaggle/input/your-dataset/best_model.pt"
model, optimizer, start_epoch = resume_from_checkpoint(checkpoint_path, model, device)
```

## Key Points:

1. **Upload Checkpoint**: Upload your checkpoint as a Kaggle dataset first
2. **Update Paths**: Change all file paths to Kaggle format (`/kaggle/input/dataset-name/`)
3. **Training Mode**: Always call `model.train()` after loading
4. **Save New Checkpoints**: Save to `/kaggle/working/` directory
5. **Lower Learning Rate**: Consider using a lower learning rate for fine-tuning

## Common Issues:

- **"module." prefix**: Remove with the DataParallel handling code above
- **File not found**: Check your dataset name and file paths
- **CUDA errors**: Make sure to use `map_location=device` when loading
- **Model still in eval mode**: Always call `model.train()` explicitly

## Testing Your Setup:

Add this cell to verify everything is working:

```python
# Verify training mode
print(f"Model training mode: {model.training}")
print(f"Requires grad: {next(model.parameters()).requires_grad}")

# Test forward pass
dummy_input = torch.randn(1, 100, 80).to(device)  # Adjust shape as needed
with torch.no_grad():
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

print("✅ Model ready for training!")
```