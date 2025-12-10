"""
Training script for custom Konkani-English translation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.konkani_custom_translator import create_custom_translation_model


class TranslationDataset(Dataset):
    """Dataset for translation pairs"""
    
    def __init__(self, data_path, src_tokenizer, tgt_tokenizer, max_len=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pair = self.data[idx]
        
        # Tokenize
        src_tokens = self.src_tokenizer.encode(pair['konkani'])[:self.max_len]
        tgt_tokens = self.tgt_tokenizer.encode(pair['english'])[:self.max_len]
        
        # Pad
        src_tokens = src_tokens + [0] * (self.max_len - len(src_tokens))
        tgt_tokens = tgt_tokens + [0] * (self.max_len - len(tgt_tokens))
        
        return {
            'src': torch.tensor(src_tokens),
            'tgt': torch.tensor(tgt_tokens)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # Create masks
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        
        # Loss
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--src_vocab_size', type=int, default=5000)
    parser.add_argument('--tgt_vocab_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_custom_translation_model(
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # TODO: Load actual tokenizers
    # For now, using placeholder
    src_tokenizer = None  # Load your Konkani tokenizer
    tgt_tokenizer = None  # Load your English tokenizer
    
    # Create datasets
    # train_dataset = TranslationDataset(args.train_data, src_tokenizer, tgt_tokenizer)
    # val_dataset = TranslationDataset(args.val_data, src_tokenizer, tgt_tokenizer)
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # val_loss = validate(model, val_loader, criterion, device)
        
        # print(f"Train Loss: {train_loss:.4f}")
        # print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #     }, f"{args.save_dir}/best_translation_model.pt")
        #     print("✅ Saved best model")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
