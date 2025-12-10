#!/usr/bin/env python3
"""
Fine-tune NLLB on Konkani-English translation data
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
try:
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
except ImportError:
    # Fallback for older transformers versions
    from transformers import TrainingArguments as Seq2SeqTrainingArguments
    from transformers import Trainer as Seq2SeqTrainer
import json
from pathlib import Path
from tqdm import tqdm
import sys


class KonkaniTranslationDataset(Dataset):
    """Dataset for NLLB fine-tuning"""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = "kok_Deva"
        self.tgt_lang = "eng_Latn"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Set source language
        self.tokenizer.src_lang = self.src_lang
        
        # Tokenize source (Konkani)
        source = self.tokenizer(
            item['konkani'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (English)
        with self.tokenizer.as_target_tokenizer():
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


def compute_metrics(eval_preds):
    """Compute metrics for evaluation"""
    predictions, labels = eval_preds
    
    # Simple accuracy: count exact matches
    predictions = predictions.argmax(axis=-1)
    
    # Mask padding tokens
    mask = labels != -100
    correct = (predictions == labels) & mask
    
    accuracy = correct.sum() / mask.sum()
    
    return {'accuracy': float(accuracy)}


def finetune_nllb(
    model_name="facebook/nllb-200-distilled-600M",
    train_data="data/nllb_finetuning/train.json",
    val_data="data/nllb_finetuning/val.json",
    output_dir="checkpoints/nllb_finetuned",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    device=None
):
    """Fine-tune NLLB model"""
    
    print("="*70)
    print("FINE-TUNING NLLB FOR KONKANI")
    print("="*70)
    
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("✓ Using Mac GPU (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("✓ Using NVIDIA GPU")
        else:
            device = "cpu"
            print("✓ Using CPU")
    
    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KonkaniTranslationDataset(train_data, tokenizer)
    val_dataset = KonkaniTranslationDataset(val_data, tokenizer)
    
    print(f"✓ Train: {len(train_dataset)} pairs")
    print(f"✓ Val: {len(val_dataset)} pairs")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=50,
        fp16=False,  # Disable for Mac MPS
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING FINE-TUNING")
    print("="*70)
    print(f"\nEpochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {output_dir}")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*70)
    print("SAVING FINE-TUNED MODEL")
    print("="*70)
    
    final_output = Path(output_dir) / "final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    
    print(f"\n✓ Model saved to: {final_output}")
    
    # Evaluate
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    eval_results = trainer.evaluate()
    
    print("\nResults:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    return trainer, eval_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune NLLB for Konkani')
    parser.add_argument('--model', type=str, default='facebook/nllb-200-distilled-600M',
                       help='Base NLLB model')
    parser.add_argument('--train', type=str, default='data/nllb_finetuning/train.json',
                       help='Training data path')
    parser.add_argument('--val', type=str, default='data/nllb_finetuning/val.json',
                       help='Validation data path')
    parser.add_argument('--output', type=str, default='checkpoints/nllb_finetuned',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Check data exists
    if not Path(args.train).exists():
        print(f"\n❌ Training data not found: {args.train}")
        print("\nRun first: python scripts/prepare_nllb_training_data.py")
        return
    
    # Fine-tune
    trainer, results = finetune_nllb(
        model_name=args.model,
        train_data=args.train,
        val_data=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE! 🎉")
    print("="*70)
    print("\nTest your fine-tuned model:")
    print(f"  python scripts/test_finetuned_nllb.py --model {args.output}/final")


if __name__ == '__main__':
    main()
