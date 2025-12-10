#!/usr/bin/env python3
"""
Deep Training Diagnosis - Find Root Cause of Poor ASR Performance
================================================================
When you have enough data (10k+) but still get terrible accuracy (6%),
the problem is usually one of these fundamental issues.
"""
import torch
import torchaudio
import json
import numpy as np
from pathlib import Path
import sys
import librosa
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

def deep_diagnosis():
    """Comprehensive diagnosis for models with sufficient data but poor performance"""
    
    print("="*80)
    print("DEEP ASR TRAINING DIAGNOSIS")
    print("When you have 10k+ data but still get 6% accuracy...")
    print("="*80)
    
    issues = []
    
    # 1. Audio-Text Alignment Issues
    print("\n1. AUDIO-TEXT ALIGNMENT CHECK")
    print("-" * 50)
    alignment_issues = check_audio_text_alignment()
    issues.extend(alignment_issues)
    
    # 2. Vocabulary and Tokenization Issues
    print("\n2. VOCABULARY & TOKENIZATION CHECK")
    print("-" * 50)
    vocab_issues = check_vocabulary_issues()
    issues.extend(vocab_issues)
    
    # 3. Audio Preprocessing Issues
    print("\n3. AUDIO PREPROCESSING CHECK")
    print("-" * 50)
    audio_issues = check_audio_preprocessing()
    issues.extend(audio_issues)
    
    # 4. Model Architecture Issues
    print("\n4. MODEL ARCHITECTURE CHECK")
    print("-" * 50)
    model_issues = check_model_architecture()
    issues.extend(model_issues)
    
    # 5. Training Process Issues
    print("\n5. TRAINING PROCESS CHECK")
    print("-" * 50)
    training_issues = check_training_process()
    issues.extend(training_issues)
    
    # 6. Provide targeted solutions
    print("\n6. ROOT CAUSE ANALYSIS & SOLUTIONS")
    print("-" * 50)
    provide_targeted_solutions(issues)

def check_audio_text_alignment():
    """Check if audio and text are properly aligned"""
    issues = []
    
    # Load a few samples from manifest
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    if not manifest_path.exists():
        issues.append("CRITICAL: No training manifest found")
        return issues
    
    print("Checking audio-text alignment...")
    
    samples_to_check = 5
    with open(manifest_path, 'r') as f:
        samples = [json.loads(line) for line in f][:samples_to_check]
    
    for i, sample in enumerate(samples, 1):
        audio_path = Path(sample['audio_filepath'])
        text = sample['text']
        duration = sample.get('duration', 0)
        
        print(f"\nSample {i}:")
        print(f"  Audio: {audio_path.name}")
        print(f"  Text: {text[:100]}...")
        print(f"  Duration: {duration:.2f}s")
        
        # Check if audio file exists
        if not audio_path.exists():
            issues.append(f"CRITICAL: Audio file missing: {audio_path}")
            print(f"  ❌ Audio file not found!")
            continue
        
        # Load and check audio
        try:
            if hasattr(librosa, 'load'):
                audio, sr = librosa.load(str(audio_path), sr=16000)
                actual_duration = len(audio) / sr
                
                print(f"  Actual duration: {actual_duration:.2f}s")
                
                # Check duration mismatch
                if abs(actual_duration - duration) > 1.0:  # 1 second tolerance
                    issues.append(f"WARNING: Duration mismatch in {audio_path.name}")
                    print(f"  ⚠️  Duration mismatch: {duration:.2f}s vs {actual_duration:.2f}s")
                
                # Check audio quality
                if np.max(np.abs(audio)) < 0.01:  # Very quiet audio
                    issues.append(f"WARNING: Very quiet audio: {audio_path.name}")
                    print(f"  ⚠️  Audio seems very quiet (max: {np.max(np.abs(audio)):.4f})")
                
                # Check for silence
                silence_ratio = np.sum(np.abs(audio) < 0.001) / len(audio)
                if silence_ratio > 0.8:  # 80% silence
                    issues.append(f"WARNING: Mostly silent audio: {audio_path.name}")
                    print(f"  ⚠️  Audio is {silence_ratio*100:.1f}% silent")
                
                print(f"  ✅ Audio loaded successfully")
                
        except Exception as e:
            issues.append(f"ERROR: Cannot load audio {audio_path.name}: {e}")
            print(f"  ❌ Error loading audio: {e}")
        
        # Check text quality
        if not text or len(text.strip()) == 0:
            issues.append(f"CRITICAL: Empty text for {audio_path.name}")
            print(f"  ❌ Empty transcription!")
        elif len(text) < 5:
            issues.append(f"WARNING: Very short text for {audio_path.name}")
            print(f"  ⚠️  Very short transcription ({len(text)} chars)")
        else:
            print(f"  ✅ Text looks good ({len(text)} chars)")
    
    return issues

def check_vocabulary_issues():
    """Check vocabulary and character distribution"""
    issues = []
    
    # Load vocabulary from checkpoint
    checkpoint_path = '/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt'
    
    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            vocab = checkpoint.get('vocab', {})
            
            if not vocab:
                issues.append("CRITICAL: No vocabulary in checkpoint")
                print("❌ No vocabulary found in checkpoint")
                return issues
            
            print(f"Vocabulary size: {len(vocab)}")
            
            # Analyze vocabulary composition
            special_tokens = []
            characters = []
            other_tokens = []
            
            for token, idx in vocab.items():
                if token in ['<blank>', '<unk>', '<eos>', '<pad>', '<sos>']:
                    special_tokens.append(token)
                elif len(token) == 1:
                    characters.append(token)
                else:
                    other_tokens.append(token)
            
            print(f"Special tokens ({len(special_tokens)}): {special_tokens}")
            print(f"Characters ({len(characters)}): {characters[:20]}...")
            print(f"Other tokens ({len(other_tokens)}): {other_tokens[:10]}...")
            
            # Check for issues
            if len(characters) < 20:
                issues.append("WARNING: Very limited character set")
                print("⚠️  Very few unique characters - might be missing Devanagari")
            
            if len(special_tokens) == 0:
                issues.append("CRITICAL: No special tokens (blank, unk)")
                print("❌ Missing essential special tokens")
            
            # Check character distribution in training data
            print("\nAnalyzing character distribution in training data...")
            char_counts = analyze_character_distribution()
            
            if char_counts:
                most_common = char_counts.most_common(10)
                print("Most common characters:")
                for char, count in most_common:
                    print(f"  '{char}': {count}")
                
                # Check if vocabulary matches actual data
                data_chars = set(char_counts.keys())
                vocab_chars = set(characters)
                
                missing_in_vocab = data_chars - vocab_chars
                if missing_in_vocab:
                    issues.append("CRITICAL: Characters in data missing from vocabulary")
                    print(f"❌ Characters in data but not in vocab: {list(missing_in_vocab)[:10]}")
                
                unused_in_vocab = vocab_chars - data_chars
                if len(unused_in_vocab) > len(vocab_chars) * 0.5:  # >50% unused
                    issues.append("WARNING: Many vocabulary characters unused")
                    print(f"⚠️  {len(unused_in_vocab)} vocab chars not in training data")
            
        except Exception as e:
            issues.append(f"ERROR: Cannot analyze vocabulary: {e}")
            print(f"❌ Error loading checkpoint: {e}")
    else:
        issues.append("CRITICAL: No checkpoint found for vocabulary analysis")
        print("❌ No checkpoint found")
    
    return issues

def analyze_character_distribution():
    """Analyze character distribution in training data"""
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not manifest_path.exists():
        return None
    
    char_counts = Counter()
    
    with open(manifest_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get('text', '')
            for char in text:
                char_counts[char] += 1
    
    return char_counts

def check_audio_preprocessing():
    """Check if audio preprocessing is correct"""
    issues = []
    
    print("Checking audio preprocessing pipeline...")
    
    # Test audio loading and preprocessing
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    if not manifest_path.exists():
        issues.append("CRITICAL: No training manifest")
        return issues
    
    with open(manifest_path, 'r') as f:
        sample = json.loads(f.readline())
    
    audio_path = Path(sample['audio_filepath'])
    if not audio_path.exists():
        issues.append("CRITICAL: Cannot test preprocessing - no audio file")
        return issues
    
    try:
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=16000)
        print(f"✅ Audio loaded: {len(audio)} samples at {sr}Hz")
        
        # Check sample rate
        if sr != 16000:
            issues.append("WARNING: Sample rate not 16kHz")
            print(f"⚠️  Sample rate: {sr}Hz (should be 16000Hz)")
        
        # Test mel spectrogram extraction
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        mel_spec = mel_transform(audio_tensor)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        print(f"✅ Mel spectrogram: {mel_spec.shape}")
        
        # Check for common preprocessing issues
        if torch.isnan(mel_spec).any():
            issues.append("CRITICAL: NaN values in mel spectrogram")
            print("❌ NaN values detected in preprocessing")
        
        if torch.isinf(mel_spec).any():
            issues.append("CRITICAL: Infinite values in mel spectrogram")
            print("❌ Infinite values detected in preprocessing")
        
        # Check dynamic range
        mel_min, mel_max = mel_spec.min(), mel_spec.max()
        if mel_max - mel_min < 1.0:  # Very small dynamic range
            issues.append("WARNING: Very small dynamic range in mel spectrogram")
            print(f"⚠️  Small dynamic range: {mel_min:.2f} to {mel_max:.2f}")
        
        print(f"✅ Mel spectrogram range: {mel_min:.2f} to {mel_max:.2f}")
        
    except Exception as e:
        issues.append(f"CRITICAL: Audio preprocessing failed: {e}")
        print(f"❌ Preprocessing error: {e}")
    
    return issues

def check_model_architecture():
    """Check if model architecture is appropriate"""
    issues = []
    
    print("Checking model architecture...")
    
    checkpoint_path = '/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt'
    
    if not Path(checkpoint_path).exists():
        issues.append("CRITICAL: No checkpoint to analyze")
        return issues
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Analyze model size
        total_params = 0
        for name, param in state_dict.items():
            total_params += param.numel()
        
        print(f"Total parameters: {total_params:,}")
        
        # Check model components
        has_encoder = any('encoder' in name for name in state_dict.keys())
        has_decoder = any('decoder' in name for name in state_dict.keys())
        has_ctc = any('ctc_head' in name for name in state_dict.keys())
        
        print(f"Has encoder: {has_encoder}")
        print(f"Has decoder: {has_decoder}")
        print(f"Has CTC head: {has_ctc}")
        
        if not has_ctc:
            issues.append("CRITICAL: No CTC head found")
            print("❌ Missing CTC head")
        
        # Check vocabulary size consistency
        if has_ctc:
            ctc_vocab_size = state_dict['ctc_head.weight'].shape[0]
            print(f"CTC vocabulary size: {ctc_vocab_size}")
            
            if has_decoder:
                decoder_vocab_size = state_dict['decoder.output_proj.weight'].shape[0]
                print(f"Decoder vocabulary size: {decoder_vocab_size}")
                
                if ctc_vocab_size != decoder_vocab_size:
                    issues.append("CRITICAL: CTC and decoder vocab size mismatch")
                    print("❌ Vocabulary size mismatch between CTC and decoder")
        
        # Check model size appropriateness
        if total_params > 50_000_000:  # 50M parameters
            issues.append("WARNING: Very large model for available data")
            print("⚠️  Model might be too large for your dataset")
        elif total_params < 5_000_000:  # 5M parameters
            issues.append("WARNING: Very small model")
            print("⚠️  Model might be too small for ASR task")
        else:
            print("✅ Model size seems appropriate")
        
    except Exception as e:
        issues.append(f"ERROR: Cannot analyze model architecture: {e}")
        print(f"❌ Error: {e}")
    
    return issues

def check_training_process():
    """Check training process and loss curves"""
    issues = []
    
    print("Checking training process...")
    
    # Check if we have training logs
    checkpoint_path = '/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt'
    
    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            epoch = checkpoint.get('epoch', 0)
            val_loss = checkpoint.get('val_loss', 0)
            
            print(f"Best model: Epoch {epoch}, Val Loss: {val_loss:.4f}")
            
            # Analyze loss values
            if val_loss > 4.0:
                issues.append("CRITICAL: Extremely high validation loss")
                print("❌ Validation loss is extremely high - model not learning")
            elif val_loss > 2.5:
                issues.append("WARNING: High validation loss")
                print("⚠️  Validation loss is high - poor convergence")
            
            # Check if model trained long enough
            if epoch < 50:
                issues.append("WARNING: Model might need more training")
                print(f"⚠️  Only trained for {epoch} epochs - might need more")
            
        except Exception as e:
            issues.append(f"ERROR: Cannot analyze training process: {e}")
            print(f"❌ Error: {e}")
    
    # Check for common training issues
    print("\nChecking for common training problems...")
    
    # Look for gradient explosion/vanishing indicators
    # This would require training logs, which we don't have access to
    print("⚠️  Cannot check gradient norms without training logs")
    
    return issues

def provide_targeted_solutions(issues):
    """Provide specific solutions based on identified issues"""
    
    if not issues:
        print("✅ No major issues detected!")
        print("Your model architecture and data seem fine.")
        print("The problem might be in hyperparameters or need more training time.")
        return
    
    print(f"\nIDENTIFIED {len(issues)} ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    print("\n" + "="*80)
    print("TARGETED SOLUTIONS BASED ON ROOT CAUSE:")
    print("="*80)
    
    # Categorize issues
    critical_issues = [i for i in issues if "CRITICAL" in i]
    audio_issues = [i for i in issues if "audio" in i.lower() or "preprocessing" in i.lower()]
    vocab_issues = [i for i in issues if "vocab" in i.lower() or "character" in i.lower()]
    model_issues = [i for i in issues if "model" in i.lower() or "architecture" in i.lower()]
    
    if critical_issues:
        print("\n🚨 CRITICAL ISSUES (Fix These First):")
        for issue in critical_issues:
            print(f"   • {issue}")
        
        if any("missing" in issue.lower() for issue in critical_issues):
            print("\n   SOLUTION: Fix Data Pipeline")
            print("   1. Verify all audio files exist")
            print("   2. Check manifest file paths are correct")
            print("   3. Regenerate manifests if needed:")
            print("      python scripts/prepare_raw_corpus_data.py --regenerate_manifests")
    
    if vocab_issues:
        print("\n📝 VOCABULARY ISSUES:")
        for issue in vocab_issues:
            print(f"   • {issue}")
        
        print("\n   SOLUTION: Fix Vocabulary")
        print("   1. Regenerate vocabulary from actual training data")
        print("   2. Ensure all characters in data are in vocabulary")
        print("   3. Use proper Devanagari character encoding")
        print("   4. Run: python scripts/fix_vocabulary.py")
    
    if audio_issues:
        print("\n🎵 AUDIO PREPROCESSING ISSUES:")
        for issue in audio_issues:
            print(f"   • {issue}")
        
        print("\n   SOLUTION: Fix Audio Pipeline")
        print("   1. Standardize sample rate to 16kHz")
        print("   2. Normalize audio amplitude")
        print("   3. Check mel spectrogram parameters")
        print("   4. Run: python scripts/fix_audio_preprocessing.py")
    
    if model_issues:
        print("\n🏗️  MODEL ARCHITECTURE ISSUES:")
        for issue in model_issues:
            print(f"   • {issue}")
        
        print("\n   SOLUTION: Adjust Model")
        print("   1. Ensure CTC and decoder vocab sizes match")
        print("   2. Use appropriate model size for data")
        print("   3. Consider simpler architecture first")
    
    # General recommendations
    print("\n🎯 GENERAL RECOMMENDATIONS:")
    print("   1. Start with a simpler CTC-only model (no attention decoder)")
    print("   2. Use transfer learning from Wav2Vec2 or Whisper")
    print("   3. Verify your data quality manually (listen to audio + read text)")
    print("   4. Test with a tiny dataset (100 samples) to ensure pipeline works")
    print("   5. Use a proven ASR framework like ESPnet or SpeechBrain")
    
    print("\n🚀 QUICK TEST:")
    print("   Try Whisper on your audio to see if it's a data quality issue:")
    print("   pip install openai-whisper")
    print("   python scripts/test_whisper_quick.py")

if __name__ == '__main__':
    deep_diagnosis()