#!/usr/bin/env python3
"""
Fix Vocabulary Issues in ASR Model
==================================
The main reason for 6% accuracy: vocabulary mismatch between training and inference
"""
import torch
import json
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

def fix_vocabulary_issues():
    """Fix vocabulary problems that cause poor ASR performance"""
    
    print("="*70)
    print("FIXING VOCABULARY ISSUES")
    print("="*70)
    
    # 1. Generate correct vocabulary from training data
    print("\n1. GENERATING VOCABULARY FROM TRAINING DATA")
    print("-" * 50)
    
    vocab = generate_vocabulary_from_data()
    
    if not vocab:
        print("❌ Cannot generate vocabulary - no training data found")
        return
    
    # 2. Update checkpoint with correct vocabulary
    print("\n2. UPDATING CHECKPOINT WITH CORRECT VOCABULARY")
    print("-" * 50)
    
    update_checkpoint_vocabulary(vocab)
    
    # 3. Test the fixed model
    print("\n3. TESTING FIXED MODEL")
    print("-" * 50)
    
    test_fixed_model()

def generate_vocabulary_from_data():
    """Generate vocabulary from actual training data"""
    
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not manifest_path.exists():
        print(f"❌ Training manifest not found: {manifest_path}")
        return None
    
    print(f"Reading training data from: {manifest_path}")
    
    # Count all characters in training data
    char_counter = Counter()
    total_samples = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get('text', '')
            
            for char in text:
                char_counter[char] += 1
            
            total_samples += 1
    
    print(f"Processed {total_samples} training samples")
    print(f"Found {len(char_counter)} unique characters")
    
    # Create vocabulary
    vocab = {
        '<blank>': 0,  # CTC blank token (MUST be 0)
        '<unk>': 1,    # Unknown token
    }
    
    # Add characters sorted by frequency (most common first)
    for char, count in char_counter.most_common():
        if char not in vocab:
            vocab[char] = len(vocab)
    
    print(f"Created vocabulary with {len(vocab)} tokens")
    
    # Show vocabulary statistics
    print("\\nVocabulary composition:")
    print(f"  Special tokens: 2 (<blank>, <unk>)")
    print(f"  Characters: {len(vocab) - 2}")
    
    # Show most common characters
    print("\\nMost common characters:")
    for char, count in char_counter.most_common(20):
        vocab_id = vocab.get(char, -1)
        print(f"  '{char}' (id: {vocab_id}): {count} times")
    
    # Check for potential issues
    devanagari_chars = [char for char in vocab.keys() if ord(char) >= 0x0900 and ord(char) <= 0x097F]
    latin_chars = [char for char in vocab.keys() if ord(char) >= 0x0041 and ord(char) <= 0x007A]
    
    print(f"\\nCharacter analysis:")
    print(f"  Devanagari characters: {len(devanagari_chars)}")
    print(f"  Latin characters: {len(latin_chars)}")
    print(f"  Other characters: {len(vocab) - 2 - len(devanagari_chars) - len(latin_chars)}")
    
    if len(devanagari_chars) < 20:
        print("⚠️  WARNING: Very few Devanagari characters - check text encoding")
    
    # Save vocabulary for inspection
    vocab_path = Path('outputs/generated_vocabulary.json')
    vocab_path.parent.mkdir(exist_ok=True)
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vocabulary saved to: {vocab_path}")
    
    return vocab

def update_checkpoint_vocabulary(vocab):
    """Update checkpoint with correct vocabulary"""
    
    checkpoint_path = Path('/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt')
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check current vocabulary
        old_vocab = checkpoint.get('vocab', {})
        print(f"Old vocabulary size: {len(old_vocab) if old_vocab else 'None'}")
        print(f"New vocabulary size: {len(vocab)}")
        
        # Update vocabulary
        checkpoint['vocab'] = vocab
        
        # Check if model vocab size matches
        state_dict = checkpoint.get('model_state_dict', {})
        if 'ctc_head.weight' in state_dict:
            model_vocab_size = state_dict['ctc_head.weight'].shape[0]
            print(f"Model CTC vocab size: {model_vocab_size}")
            
            if model_vocab_size != len(vocab):
                print(f"⚠️  WARNING: Model vocab size ({model_vocab_size}) != data vocab size ({len(vocab)})")
                print("   This mismatch could be causing the poor performance!")
                print("   You may need to retrain with the correct vocabulary size.")
        
        # Save updated checkpoint
        fixed_checkpoint_path = checkpoint_path.parent / 'best_model_fixed_vocab.pt'
        torch.save(checkpoint, fixed_checkpoint_path)
        
        print(f"✅ Updated checkpoint saved to: {fixed_checkpoint_path}")
        
        # Also update the original (backup first)
        backup_path = checkpoint_path.parent / 'best_model_backup.pt'
        if not backup_path.exists():
            torch.save(checkpoint, backup_path)
            print(f"✅ Backup created: {backup_path}")
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Original checkpoint updated with vocabulary")
        
    except Exception as e:
        print(f"❌ Error updating checkpoint: {e}")

def test_fixed_model():
    """Test the model with fixed vocabulary"""
    
    print("Testing model with fixed vocabulary...")
    
    # Import and run the existing test script
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'scripts/test_asr_latest.py'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        print("Test results:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        print("Please run manually: python scripts/test_asr_latest.py")

def create_whisper_test():
    """Create a quick Whisper test to compare"""
    
    whisper_script = '''#!/usr/bin/env python3
"""
Test Whisper on Konkani Audio
"""
try:
    import whisper
    from pathlib import Path
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Test on a sample
    audio_file = "data/konkani-asr-v0/data/processed_segments_diarized/audio_segments/segment_000008.wav"
    
    if Path(audio_file).exists():
        print(f"Testing Whisper on: {audio_file}")
        
        # Try with Hindi (closest to Konkani)
        result = model.transcribe(audio_file, language="hi")
        print(f"Whisper (Hindi): {result['text']}")
        
        # Try auto-detect
        result_auto = model.transcribe(audio_file)
        print(f"Whisper (auto): {result_auto['text']}")
        
    else:
        print("❌ No test audio file found")
        
except ImportError:
    print("❌ Whisper not installed. Install with: pip install openai-whisper")
except Exception as e:
    print(f"❌ Error: {e}")
'''
    
    with open('scripts/test_whisper_quick.py', 'w') as f:
        f.write(whisper_script)
    
    print("✅ Created Whisper test script: scripts/test_whisper_quick.py")

if __name__ == '__main__':
    fix_vocabulary_issues()
    create_whisper_test()
    
    print("\\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Test the fixed model:")
    print("   python scripts/test_asr_latest.py")
    print()
    print("2. Compare with Whisper:")
    print("   pip install openai-whisper")
    print("   python scripts/test_whisper_quick.py")
    print()
    print("3. If still poor performance, the issue is likely:")
    print("   - Model vocab size != data vocab size (need to retrain)")
    print("   - Training hyperparameters")
    print("   - Need transfer learning instead of training from scratch")