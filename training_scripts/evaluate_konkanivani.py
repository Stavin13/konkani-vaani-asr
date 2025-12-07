"""
Evaluate KonkaniVani ASR Model
==============================
Calculate WER and CER metrics on test set
"""

import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))

from models.konkanivani_asr import create_konkanivani_model
from data.audio_processing.dataset import create_dataloaders
from data.audio_processing.text_tokenizer import KonkaniTokenizer
import jiwer


class ASREvaluator:
    """Evaluate ASR model performance"""
    
    def __init__(self, model, tokenizer, test_loader, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    def evaluate(self):
        """
        Evaluate model on test set
        
        Returns:
            metrics: Dict with WER, CER, and other metrics
        """
        all_predictions = []
        all_references = []
        
        print(f"\n{'='*80}")
        print("EVALUATING KONKANIVANI ASR")
        print(f"{'='*80}\n")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                audio_features = batch['audio_features'].to(self.device)
                transcript_tokens = batch['transcript_tokens']
                audio_lengths = batch['audio_lengths'].to(self.device)
                
                # Forward pass (CTC only for evaluation)
                ctc_logits, _ = self.model(audio_features, audio_lengths)
                
                # CTC greedy decoding
                ctc_preds = ctc_logits.argmax(dim=-1)
                
                # Decode predictions and references
                for i in range(len(ctc_preds)):
                    # Prediction
                    pred_tokens = ctc_preds[i].cpu().tolist()
                    pred_text = self.tokenizer.decode_ctc(pred_tokens)
                    
                    # Reference
                    ref_tokens = transcript_tokens[i].cpu().tolist()
                    ref_text = self.tokenizer.decode(ref_tokens, remove_special=True)
                    
                    all_predictions.append(pred_text)
                    all_references.append(ref_text)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_references)
        
        # Print results
        self.print_metrics(metrics)
        
        # Save sample predictions
        self.save_samples(all_predictions, all_references, num_samples=20)
        
        return metrics
    
    def calculate_metrics(self, predictions, references):
        """Calculate WER and CER"""
        # Word Error Rate
        wer = jiwer.wer(references, predictions)
        
        # Character Error Rate
        cer = jiwer.cer(references, predictions)
        
        # Additional metrics
        mer = jiwer.mer(references, predictions)  # Match Error Rate
        wil = jiwer.wil(references, predictions)  # Word Information Lost
        
        return {
            'wer': wer * 100,  # Convert to percentage
            'cer': cer * 100,
            'mer': mer * 100,
            'wil': wil * 100,
            'num_samples': len(predictions)
        }
    
    def print_metrics(self, metrics):
        """Print evaluation metrics"""
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"\nWord Error Rate (WER): {metrics['wer']:.2f}%")
        print(f"Character Error Rate (CER): {metrics['cer']:.2f}%")
        print(f"Match Error Rate (MER): {metrics['mer']:.2f}%")
        print(f"Word Information Lost (WIL): {metrics['wil']:.2f}%")
        print(f"{'='*80}\n")
    
    def save_samples(self, predictions, references, num_samples=20):
        """Save sample predictions for inspection"""
        samples = []
        for i in range(min(num_samples, len(predictions))):
            samples.append({
                'reference': references[i],
                'prediction': predictions[i]
            })
        
        output_file = 'evaluation_samples.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(samples)} sample predictions to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate KonkaniVani ASR')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_manifest', type=str,
                        default='data/konkani-asr-v0/splits/manifests/test.json',
                        help='Path to test manifest')
    parser.add_argument('--vocab_file', type=str, default='vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run evaluation on')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = KonkaniTokenizer(args.vocab_file)
    
    # Create test dataloader
    from data.audio_processing.audio_processor import AudioProcessor
    from data.audio_processing.dataset import KonkaniASRDataset
    from torch.utils.data import DataLoader
    from data.audio_processing.dataset import collate_fn
    
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=80, spec_augment=False)
    test_dataset = KonkaniASRDataset(
        args.test_manifest,
        tokenizer,
        audio_processor,
        apply_augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id)
    )
    
    # Load model
    model = create_konkanivani_model(vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded model from {args.checkpoint}")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    
    # Evaluate
    evaluator = ASREvaluator(model, tokenizer, test_loader, device)
    metrics = evaluator.evaluate()
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
