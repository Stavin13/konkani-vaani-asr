import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import json
import os
import math
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# CONFIGURATION
CONFIG = {
    "corpus_path": "data/konkani_corpus_for_lm.txt",
    "spm_model": "data/bpe_tokenizer/konkani_bpe.model",
    "output_dir": "models/language_models/neural_lm",
    "batch_size": 32,
    "max_len": 128,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "lr": 5e-4,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_model*4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # x: (B, T)
        T = x.size(1)
        mask = self.generate_square_subsequent_mask(T).to(x.device)
        
        x = self.embedding(x) + self.pos_encoder[:, :T, :]
        x = self.transformer_decoder(x, x, tgt_mask=mask, memory_mask=None)
        logits = self.fc_out(x)
        return logits

class KonkaniTextDataset(Dataset):
    def __init__(self, corpus_path, spm_model_path, max_len):
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        self.max_len = max_len
        
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"Tokenizing {len(lines)} sentences...")
        self.tokenized_data = []
        for line in tqdm(lines):
            ids = self.sp.encode(line.strip())
            # Truncate and wrap with SOS/EOS if helpful, but for LM we can just pad
            if len(ids) > self.max_len - 1:
                ids = ids[:self.max_len-1]
            
            # Input: [SOS, token1, ..., tokenN]
            # Target: [token1, ..., tokenN, EOS]
            # For simplicity, we just use the line and shift during loss
            self.tokenized_data.append(torch.LongTensor(ids))

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        ids = self.tokenized_data[idx]
        return ids

def collate_fn(batch):
    # Pad sequences to the max length in the batch
    max_batch_len = max(len(ids) for ids in batch)
    if max_batch_len < 2: max_batch_len = 2 # Need at least 2 for input/target shift
    
    padded_batch = torch.zeros(len(batch), max_batch_len, dtype=torch.long)
    for i, ids in enumerate(batch):
        padded_batch[i, :len(ids)] = ids
    return padded_batch

def train():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # Load dataset
    dataset = KonkaniTextDataset(CONFIG['corpus_path'], CONFIG['spm_model'], CONFIG['max_len'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    vocab_size = dataset.sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")
    
    model = TransformerLM(vocab_size, CONFIG['d_model'], CONFIG['nhead'], CONFIG['num_layers'], CONFIG['max_len']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Index 0 is padding
    
    print(f"Starting training on {device}...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            batch = batch.to(device)
            if batch.size(1) < 2: continue
            
            # LM setup: Predict next token
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            outputs = model(input_ids)
            # Reshape for cross entropy
            loss = criterion(outputs.reshape(-1, vocab_size), target_ids.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else 999
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            'vocab_size': vocab_size,
            'config': CONFIG
        }, os.path.join(CONFIG['output_dir'], f"konkani_transformer_lm_ep{epoch+1}.pt"))

if __name__ == "__main__":
    train()
