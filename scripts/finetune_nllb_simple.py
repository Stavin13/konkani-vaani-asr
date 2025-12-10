#!/usr/bin/env python3
"""
Simple fine-tuning for NLLB without Seq2SeqTrainer
Uses manual training loop for compatibility
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from pathlib import Path
from tqdm import tqdm


class KonkaniDataset(Dataset):
    """Simple dataset for NLLB fine-tuning"""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize source (Konkani)
        self.tokenizer.src_lang = "kok_Deva"
        source = self.tokenizer(
            item['konkani'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (English)
        target = self.tokenizer(
            item['english'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Replace padding token id with -100 (ignored in loss)
        labels[labels == 0] = -100
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            labels[labels == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def finetune_nllb(
    model_name="facebook/nllb-200-distilled-600M",
    train_data="data/nllb_finetuning/train.json",
    val_data="data/nllb_finetuning/val.json",
    output_dir="checkpoints/nllb_finetuned",
    num_epochs=10,
    batch_size=4,
    learning_rate=2e-5
):
    """Fine-tune NLLB model"""
    
    print("="*70)
    print("FINE-TUNING NLLB FOR KONKANI")
    print("="*70)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KonkaniDataset(train_data, tokenizer)
    val_dataset = KonkaniDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"✓ Train: {len(train_dataset)} pairs")
    print(f"✓ Val: {len(val_dataset)} pairs")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training
    print("\n" + "="*70)
    print("STARTING FINE-TUNING")
    print("="*70)
    print(f"\nEpochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    best_val_loss = float('inf')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            final_path = output_path / "final"
            final_path.mkdir(exist_ok=True)
            
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        checkpoint_path = output_path / f"checkpoint-epoch-{epoch}"
        checkpoint_path.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE! 🎉")
    print("="*70)
    print(f"\nBest model saved to: {output_path / 'final'}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    return model, tokenizer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune NLLB for Konkani')
    parser.add_argument('--model', type=str, default='facebook/nllb-200-distilled-600M')
    parser.add_argument('--train', type=str, default='data/nllb_finetuning/train.json')
    parser.add_argument('--val', type=str, default='data/nllb_finetuning/val.json')
    parser.add_argument('--output', type=str, default='checkpoints/nllb_finetuned')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    
    # Check data exists
    if not Path(args.train).exists():
        print(f"\n❌ Training data not found: {args.train}")
        print("\nRun first: python scripts/prepare_nllb_training_data.py")
        return
    
    # Fine-tune
    model, tokenizer = finetune_nllb(
        model_name=args.model,
        train_data=args.train,
        val_data=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\nTest your fine-tuned model:")
    print(f"  python scripts/test_finetuned_nllb.py --model {args.output}/final --mode test")


if __name__ == '__main__':
    main()
