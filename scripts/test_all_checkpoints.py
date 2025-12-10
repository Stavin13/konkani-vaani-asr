#!/usr/bin/env python3
"""
Test All Checkpoints in Kaggle Downloads
========================================
Compare performance of all available checkpoints to find the best one
"""
import torch
import torchaudio
from pathlib import Path
import sys
import json
import numpy as np
from collections import defaultdict
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import librosa as fallback
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from models.konkanivani_asr import KonkaniVaniASR

def test_all_checkpoints():
    """Test all checkpoints in the kaggle downloads directory"""
    
    print("="*80)
    print("TESTING ALL CHECKPOINTS")
    print("="*80)
    
    # Find all checkpoints
    checkpoint_dir = Path('/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints')
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Get all .pt files
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    checkpoints = [cp for cp in checkpoints if not cp.name.startswith('.')]  # Skip hidden files
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in sorted(checkpoints):
        print(f"  - {cp.name}")
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    # Load test samples
    test_samples = load_test_samples()
    if not test_samples:
        print("❌ No test samples found!")
        return
    
    print(f"\nLoaded {len(test_samples)} test samples")
    
    # Test each checkpoint
    results = {}
    
    for checkpoint_path in sorted(checkpoints):
        print(f"\n{'='*80}")
        print(f"TESTING: {checkpoint_path.name}")
        print(f"{'='*80}")
        
        try:
            result = test_single_checkpoint(checkpoint_path, test_samples)
            results[checkpoint_path.name] = result
            
        except Exception as e:
            print(f"❌ Error testing {checkpoint_path.name}: {e}")
            results[checkpoint_path.name] = {
                'error': str(e),
                'epoch': 'unknown',
                'val_loss': float('inf'),
                'accuracy': 0.0
            }
    
    # Compare results
    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON")
    print(f"{'='*80}")
    
    compare_checkpoints(results)

def load_test_samples(max_samples=5):
    """Load test samples for evaluation"""
    
    test_manifest = Path('data/konkani-asr-v0/splits/manifests/test.json')
    
    if not test_manifest.exists():
        print(f"❌ Test manifest not found: {test_manifest}")
        return []
    
    samples = []
    with open(test_manifest, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            data = json.loads(line)
            audio_path = Path(data['audio_filepath'])
            
            if audio_path.exists():
                samples.append(data)
            else:
                print(f"⚠️  Audio not found: {audio_path}")
    
    return samples

def test_single_checkpoint(checkpoint_path, test_samples):
    """Test a single checkpoint on test samples"""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    
    print(f"  Epoch: {epoch}")
    print(f"  Val Loss: {val_loss}")
    
    # Infer model architecture
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    vocab_size = state_dict['ctc_head.weight'].shape[0]
    d_model = state_dict['encoder.input_proj.weight'].shape[0]
    
    print(f"  Vocab Size: {vocab_size}")
    print(f"  D Model: {d_model}")
    
    # Create and load model
    model = KonkaniVaniASR(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=12,
        dropout=0.1
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load vocabulary (if available)
    vocab = checkpoint.get('vocab', None)
    if vocab is None:
        vocab = load_vocab_from_manifest()
    
    print(f"  Vocabulary: {len(vocab) if vocab else 'None'}")
    
    # Test on samples
    print(f"\nTesting on {len(test_samples)} samples:")
    
    accuracies = []
    predictions = []
    
    for i, sample in enumerate(test_samples, 1):
        audio_path = Path(sample['audio_filepath'])
        true_text = sample['text']
        
        print(f"\n[{i}] {audio_path.name}")
        print(f"  True: {true_text[:60]}...")
        
        try:
            predicted_text = transcribe_audio(model, audio_path, vocab)
            
            if predicted_text:
                print(f"  Pred: {predicted_text[:60]}...")
                
                # Calculate character accuracy
                accuracy = calculate_character_accuracy(true_text, predicted_text)
                accuracies.append(accuracy)
                
                print(f"  Accuracy: {accuracy:.1f}%")
            else:
                print(f"  Pred: [FAILED]")
                accuracies.append(0.0)
            
            predictions.append({
                'audio': audio_path.name,
                'true': true_text,
                'pred': predicted_text or '[FAILED]',
                'accuracy': accuracies[-1] if accuracies else 0.0
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            accuracies.append(0.0)
            predictions.append({
                'audio': audio_path.name,
                'true': true_text,
                'pred': f'[ERROR: {e}]',
                'accuracy': 0.0
            })
    
    # Calculate overall metrics
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    
    print(f"\nOverall Results:")
    print(f"  Average Accuracy: {avg_accuracy:.1f}%")
    print(f"  Successful Predictions: {sum(1 for a in accuracies if a > 0)}/{len(accuracies)}")
    
    return {
        'epoch': epoch,
        'val_loss': val_loss,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'avg_accuracy': avg_accuracy,
        'successful_predictions': sum(1 for a in accuracies if a > 0),
        'total_predictions': len(accuracies),
        'predictions': predictions
    }

def transcribe_audio(model, audio_path, vocab):
    """Transcribe audio file using the model"""
    
    try:
        # Load audio
        if HAS_LIBROSA:
            audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = mel_spec.transpose(1, 2)  # (batch, time, features)
        
        # Forward pass
        with torch.no_grad():
            output = model(mel_spec)
            if isinstance(output, tuple):
                logits = output[0]  # CTC logits
            else:
                logits = output
        
        # Decode predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Convert to text
        if vocab:
            reverse_vocab = {v: k for k, v in vocab.items()}
            
            text = []
            prev_idx = None
            for idx in predictions[0].tolist():
                if idx != 0 and idx != prev_idx:  # 0 is blank
                    char = reverse_vocab.get(idx, '<unk>')
                    if char not in ['<blank>', '<unk>']:
                        text.append(char)
                prev_idx = idx
            
            return ''.join(text)
        else:
            return f"[No vocab - raw predictions: {predictions[0].tolist()[:20]}...]"
    
    except Exception as e:
        return None

def calculate_character_accuracy(true_text, pred_text):
    """Calculate character-level accuracy"""
    
    if not pred_text or not true_text:
        return 0.0
    
    # Simple character-level accuracy
    correct = 0
    total = max(len(true_text), len(pred_text))
    
    for i in range(min(len(true_text), len(pred_text))):
        if true_text[i] == pred_text[i]:
            correct += 1
    
    return 100.0 * correct / total if total > 0 else 0.0

def load_vocab_from_manifest():
    """Load vocabulary from training manifest"""
    
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not manifest_path.exists():
        return None
    
    vocab = {'<blank>': 0, '<unk>': 1}
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            for char in text:
                if char not in vocab:
                    vocab[char] = len(vocab)
    
    return vocab

def compare_checkpoints(results):
    """Compare all checkpoint results"""
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Sort by accuracy
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1].get('avg_accuracy', 0), 
        reverse=True
    )
    
    print(f"{'Rank':<4} {'Checkpoint':<25} {'Epoch':<6} {'Val Loss':<10} {'Accuracy':<10} {'Success':<8} {'Status'}")
    print("-" * 80)
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        if 'error' in result:
            status = f"❌ {result['error'][:20]}..."
            epoch = result.get('epoch', 'N/A')
            val_loss = 'N/A'
            accuracy = 'N/A'
            success = 'N/A'
        else:
            epoch = result.get('epoch', 'N/A')
            val_loss = f"{result.get('val_loss', 0):.4f}" if isinstance(result.get('val_loss'), (int, float)) else 'N/A'
            accuracy = f"{result.get('avg_accuracy', 0):.1f}%"
            success = f"{result.get('successful_predictions', 0)}/{result.get('total_predictions', 0)}"
            
            if result.get('avg_accuracy', 0) > 10:
                status = "✅ Good"
            elif result.get('avg_accuracy', 0) > 5:
                status = "⚠️  Poor"
            else:
                status = "❌ Bad"
        
        print(f"{rank:<4} {name:<25} {epoch:<6} {val_loss:<10} {accuracy:<10} {success:<8} {status}")
    
    # Find best checkpoint
    best_checkpoint = sorted_results[0]
    best_name, best_result = best_checkpoint
    
    print(f"\n🏆 BEST CHECKPOINT: {best_name}")
    
    if 'error' not in best_result:
        print(f"   Epoch: {best_result.get('epoch', 'N/A')}")
        print(f"   Validation Loss: {best_result.get('val_loss', 'N/A')}")
        print(f"   Average Accuracy: {best_result.get('avg_accuracy', 0):.1f}%")
        print(f"   Vocab Size: {best_result.get('vocab_size', 'N/A')}")
        
        # Show sample predictions from best model
        if 'predictions' in best_result:
            print(f"\n📝 SAMPLE PREDICTIONS FROM BEST MODEL:")
            for i, pred in enumerate(best_result['predictions'][:3], 1):
                print(f"\n[{i}] {pred['audio']}")
                print(f"    True: {pred['true'][:80]}...")
                print(f"    Pred: {pred['pred'][:80]}...")
                print(f"    Accuracy: {pred['accuracy']:.1f}%")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    best_accuracy = best_result.get('avg_accuracy', 0) if 'error' not in best_result else 0
    
    if best_accuracy < 5:
        print("   ❌ All models perform very poorly (< 5% accuracy)")
        print("   → The vocabulary mismatch issue affects ALL checkpoints")
        print("   → You need to retrain with vocab_size=193")
        print("   → Or use transfer learning (Whisper/Wav2Vec2)")
    elif best_accuracy < 15:
        print("   ⚠️  Best model still has poor accuracy (< 15%)")
        print("   → Vocabulary mismatch is likely the main issue")
        print("   → Consider retraining with correct vocabulary size")
    else:
        print("   ✅ Best model shows reasonable performance")
        print("   → Use this checkpoint for further development")
        print("   → Consider fine-tuning or more training")
    
    # Save detailed results
    results_path = Path('outputs/checkpoint_comparison_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Detailed results saved to: {results_path}")

if __name__ == '__main__':
    test_all_checkpoints()  