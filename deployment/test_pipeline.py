"""
Quick test script for the pipeline
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        import torch
        print("✅ PyTorch")
        
        import torchaudio
        print("✅ TorchAudio")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("✅ Transformers")
        
        import streamlit
        print("✅ Streamlit")
        
        # Test model imports
        sys.path.append(str(Path(__file__).parent.parent))
        from models.konkanivani_asr import KonkaniVaniASR
        print("✅ ASR Model")
        
        from models.konkani_ner import KonkaniNER
        print("✅ NER Model")
        
        from models.konkani_custom_emotion import CustomEmotionModel
        print("✅ Emotion Model")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_checkpoints():
    """Test if checkpoint files exist"""
    print("\nChecking checkpoint files...")
    
    checkpoints = {
        'ASR': '../kaggle_best_model/checkpoints/best_model.pt',
        'NER': '../checkpoints/ner/best_ner_model.pt',
        'Emotion': '../checkpoints/emotion_model/emotion_model_mac.pt',
        'Translation (finetuned)': '../checkpoints/nllb_finetuned/final',
    }
    
    all_exist = True
    for name, path in checkpoints.items():
        full_path = Path(__file__).parent / path
        if full_path.exists():
            print(f"✅ {name}: {path}")
        else:
            if name == 'Translation (finetuned)':
                print(f"⚠️  {name}: {path} (will use base NLLB)")
            else:
                print(f"❌ {name}: {path} NOT FOUND")
                all_exist = False
    
    return all_exist


def test_device():
    """Test available devices"""
    print("\nChecking available devices...")
    
    import torch
    
    if torch.backends.mps.is_available():
        print("✅ Mac GPU (MPS) available")
    elif torch.cuda.is_available():
        print("✅ NVIDIA GPU (CUDA) available")
    else:
        print("ℹ️  CPU only (no GPU detected)")


def main():
    print("="*70)
    print("KONKANIVANI PIPELINE TEST")
    print("="*70)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Install dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    # Test checkpoints
    if not test_checkpoints():
        print("\n⚠️  Some checkpoints missing. Pipeline may not work fully.")
    
    # Test device
    test_device()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nTo run the app:")
    print("  streamlit run app.py")
    print("\nOr use the quick start script:")
    print("  ./run.sh")


if __name__ == '__main__':
    main()
