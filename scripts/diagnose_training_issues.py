#!/usr/bin/env python3
"""
Diagnose ASR Training Issues
============================
Analyze why your model has poor accuracy and provide solutions
"""
import torch
import json
from pathlib import Path
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

def diagnose_model_issues():
    """Comprehensive diagnosis of ASR training issues"""
    
    print("="*80)
    print("ASR TRAINING DIAGNOSIS")
    print("="*80)
    
    # 1. Check data quantity
    print("\n1. DATA QUANTITY ANALYSIS")
    print("-" * 40)
    
    data_issues = check_data_quantity()
    
    # 2. Check model predictions
    print("\n2. MODEL PREDICTION ANALYSIS")
    print("-" * 40)
    
    prediction_issues = analyze_model_predictions()
    
    # 3. Check training configuration
    print("\n3. TRAINING CONFIGURATION ANALYSIS")
    print("-" * 40)
    
    config_issues = analyze_training_config()
    
    # 4. Provide solutions
    print("\n4. RECOMMENDED SOLUTIONS")
    print("-" * 40)
    
    provide_solutions(data_issues, prediction_issues, config_issues)

def check_data_quantity():
    """Check if we have enough training data"""
    issues = []
    
    # Check manifest files
    manifests = {
        'train': 'data/konkani-asr-v0/splits/manifests/train.json',
        'val': 'data/konkani-asr-v0/splits/manifests/val.json',
        'test': 'data/konkani-asr-v0/splits/manifests/test.json'
    }
    
    total_samples = 0
    total_duration = 0
    
    for split, manifest_path in manifests.items():
        if Path(manifest_path).exists():
            with open(manifest_path, 'r') as f:
                samples = [json.loads(line) for line in f]
                duration = sum(s.get('duration', 0) for s in samples)
                
                print(f"{split.upper()}: {len(samples)} samples, {duration/3600:.1f} hours")
                total_samples += len(samples)
                total_duration += duration
        else:
            print(f"{split.upper()}: ❌ Not found")
            issues.append(f"Missing {split} manifest")
    
    print(f"\nTOTAL: {total_samples} samples, {total_duration/3600:.1f} hours")
    
    # Evaluate data sufficiency
    if total_samples < 5000:
        issues.append("CRITICAL: Too few samples (need 10k+ minimum)")
        print("❌ CRITICAL: Insufficient data for ASR training")
    elif total_samples < 10000:
        issues.append("WARNING: Limited samples (recommend 20k+)")
        print("⚠️  WARNING: Limited data, expect poor performance")
    else:
        print("✅ Data quantity looks good")
    
    if total_duration < 10:  # 10 hours
        issues.append("CRITICAL: Too little audio (need 20+ hours)")
        print("❌ CRITICAL: Insufficient audio duration")
    elif total_duration < 50:  # 50 hours
        issues.append("WARNING: Limited audio (recommend 100+ hours)")
        print("⚠️  WARNING: Limited audio duration")
    else:
        print("✅ Audio duration looks good")
    
    return issues

def analyze_model_predictions():
    """Analyze what the model is actually predicting"""
    issues = []
    
    # Load best checkpoint
    checkpoint_path = '/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt'
    
    if not Path(checkpoint_path).exists():
        issues.append("Cannot find best model checkpoint")
        print("❌ Cannot analyze predictions - no checkpoint found")
        return issues
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check training metrics
        val_loss = checkpoint.get('val_loss', 0)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Best model: Epoch {epoch}, Val Loss: {val_loss:.4f}")
        
        if val_loss > 3.0:
            issues.append("CRITICAL: Very high validation loss (>3.0)")
            print("❌ CRITICAL: Model hasn't learned properly")
        elif val_loss > 2.0:
            issues.append("WARNING: High validation loss (>2.0)")
            print("⚠️  WARNING: Model performance is poor")
        else:
            print("✅ Validation loss looks reasonable")
        
        # Analyze vocabulary
        vocab = checkpoint.get('vocab', {})
        if vocab:
            print(f"Vocabulary size: {len(vocab)}")
            
            # Check for special tokens
            special_tokens = ['<blank>', '<unk>', '<eos>', '<pad>']
            found_special = [token for token in special_tokens if token in vocab]
            print(f"Special tokens: {found_special}")
            
            # Check character distribution
            chars = [k for k in vocab.keys() if k not in special_tokens and len(k) == 1]
            print(f"Character tokens: {len(chars)}")
            
            if len(chars) < 30:
                issues.append("WARNING: Very limited character set")
                print("⚠️  WARNING: Limited character vocabulary")
        else:
            issues.append("No vocabulary found in checkpoint")
            print("❌ No vocabulary information available")
            
    except Exception as e:
        issues.append(f"Error loading checkpoint: {e}")
        print(f"❌ Error analyzing checkpoint: {e}")
    
    return issues

def analyze_training_config():
    """Analyze training configuration for common issues"""
    issues = []
    
    # Check if we have training logs or config
    config_files = [
        'config/training_config_from_checkpoint15.yaml',
        'kaggle_downloads/20251210_060024/checkpoints/best_model.pt'
    ]
    
    config_found = False
    for config_file in config_files:
        if Path(config_file).exists():
            config_found = True
            print(f"Found config: {config_file}")
            
            if config_file.endswith('.pt'):
                # Extract config from checkpoint
                try:
                    checkpoint = torch.load(config_file, map_location='cpu')
                    config = checkpoint.get('config', {})
                    if config:
                        analyze_config_dict(config, issues)
                except Exception as e:
                    print(f"Error reading config from checkpoint: {e}")
            break
    
    if not config_found:
        issues.append("No training configuration found")
        print("❌ Cannot analyze training configuration")
    
    return issues

def analyze_config_dict(config, issues):
    """Analyze configuration dictionary"""
    
    # Check learning rate
    lr = config.get('learning_rate', 0)
    if lr > 0.001:
        issues.append("WARNING: Learning rate might be too high")
        print(f"⚠️  Learning rate: {lr} (might be too high)")
    elif lr < 0.00001:
        issues.append("WARNING: Learning rate might be too low")
        print(f"⚠️  Learning rate: {lr} (might be too low)")
    else:
        print(f"✅ Learning rate: {lr}")
    
    # Check other parameters
    batch_size = config.get('batch_size', 0)
    if batch_size < 8:
        issues.append("WARNING: Batch size might be too small")
        print(f"⚠️  Batch size: {batch_size} (might be too small)")
    else:
        print(f"✅ Batch size: {batch_size}")

def provide_solutions(data_issues, prediction_issues, config_issues):
    """Provide specific solutions based on identified issues"""
    
    all_issues = data_issues + prediction_issues + config_issues
    
    if not all_issues:
        print("✅ No major issues detected!")
        print("Your model might just need more training time.")
        return
    
    print("IDENTIFIED ISSUES:")
    for i, issue in enumerate(all_issues, 1):
        print(f"{i}. {issue}")
    
    print("\nRECOMMENDED SOLUTIONS:")
    
    # Data-related solutions
    if any("Too few samples" in issue or "Too little audio" in issue for issue in all_issues):
        print("\n🔥 SOLUTION 1: GET MORE DATA (HIGHEST PRIORITY)")
        print("   Your model needs significantly more training data.")
        print("   Options:")
        print("   a) Process more audio from KonkaniRawSpeechCorpus")
        print("   b) Use data augmentation to multiply your dataset")
        print("   c) Find additional Konkani audio sources")
        print("   d) Use transfer learning (recommended)")
        
        print("\n   Quick commands:")
        print("   # Check for more raw data")
        print("   find KonkaniRawSpeechCorpus -name '*.wav' | wc -l")
        print("   # Process all available data")
        print("   python scripts/prepare_raw_corpus_data.py --process_all")
    
    # Model performance solutions
    if any("high validation loss" in issue.lower() for issue in all_issues):
        print("\n🚀 SOLUTION 2: USE TRANSFER LEARNING")
        print("   Instead of training from scratch, fine-tune a pretrained model:")
        print("   - Wav2Vec2 (Facebook's pretrained speech model)")
        print("   - Whisper (OpenAI's multilingual model)")
        print("   - These already understand speech patterns")
        
        print("\n   Implementation:")
        print("   # Install transformers")
        print("   pip install transformers")
        print("   # Use pretrained Wav2Vec2")
        print("   python scripts/finetune_wav2vec2.py")
    
    # Configuration solutions
    if any("Learning rate" in issue for issue in all_issues):
        print("\n⚙️  SOLUTION 3: FIX TRAINING CONFIGURATION")
        print("   Adjust hyperparameters:")
        print("   - Learning rate: 0.0001 - 0.001")
        print("   - Batch size: 16-32 (if GPU allows)")
        print("   - Add data augmentation")
        print("   - Use label smoothing")
        print("   - Increase training epochs to 200+")
    
    # Quick fix solution
    print("\n⚡ SOLUTION 4: QUICK TEST WITH PRETRAINED MODEL")
    print("   Test if Whisper works on your Konkani audio:")
    print("   pip install openai-whisper")
    print("   whisper your_audio.wav --model medium --language hi")
    print("   (Use Hindi as closest language)")
    
    # Data augmentation solution
    print("\n📈 SOLUTION 5: DATA AUGMENTATION")
    print("   Multiply your dataset 5-10x with augmentation:")
    print("   - Speed perturbation (0.9x, 1.1x)")
    print("   - Add background noise")
    print("   - Pitch shifting")
    print("   - SpecAugment (time/frequency masking)")
    
    print("\n" + "="*80)
    print("PRIORITY ORDER:")
    print("1. Try pretrained Whisper/Wav2Vec2 (fastest to test)")
    print("2. Get more training data (best long-term solution)")
    print("3. Use data augmentation (if can't get more data)")
    print("4. Fix training hyperparameters")
    print("="*80)

def create_quick_fix_script():
    """Create a script to test Whisper on Konkani audio"""
    
    script_content = '''#!/usr/bin/env python3
"""
Quick test: Does Whisper work on Konkani?
"""
import whisper
import sys
from pathlib import Path

def test_whisper_on_konkani():
    """Test Whisper on a Konkani audio sample"""
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Find a test audio file
    test_audio = None
    audio_paths = [
        "data/konkani-asr-v0/data/processed_segments_diarized/audio_segments/segment_000008.wav",
        "KonkaniRawSpeechCorpus/Data/Audio/konkani_001.wav"
    ]
    
    for path in audio_paths:
        if Path(path).exists():
            test_audio = path
            break
    
    if not test_audio:
        print("❌ No test audio found")
        return
    
    print(f"Testing on: {test_audio}")
    
    # Transcribe
    result = model.transcribe(test_audio, language="hi")  # Hindi as closest
    
    print("\\nWhisper Result:")
    print(f"Text: {result['text']}")
    print(f"Language detected: {result.get('language', 'unknown')}")
    
    # Also try without language hint
    result_auto = model.transcribe(test_audio)
    print("\\nWhisper (auto-detect):")
    print(f"Text: {result_auto['text']}")
    print(f"Language detected: {result_auto.get('language', 'unknown')}")

if __name__ == "__main__":
    test_whisper_on_konkani()
'''
    
    with open('scripts/test_whisper_quick.py', 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ Created quick test script: scripts/test_whisper_quick.py")
    print("Run with: python scripts/test_whisper_quick.py")

if __name__ == '__main__':
    diagnose_model_issues()
    create_quick_fix_script()