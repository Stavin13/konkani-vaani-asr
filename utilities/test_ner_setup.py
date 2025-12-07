"""
Quick test to verify NER setup is working
Tests auto-labeling on a few samples
"""

import json
import sys

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("  ✅ torch")
    except ImportError:
        print("  ❌ torch - run: pip install torch")
        return False
    
    try:
        import transformers
        print("  ✅ transformers")
    except ImportError:
        print("  ❌ transformers - run: pip install transformers")
        return False
    
    try:
        from torchcrf import CRF
        print("  ✅ pytorch-crf (optional)")
    except ImportError:
        print("  ⚠️  pytorch-crf not installed (optional)")
        print("     Model will work without CRF, just slightly lower accuracy")
    
    try:
        from tqdm import tqdm
        print("  ✅ tqdm")
    except ImportError:
        print("  ❌ tqdm - run: pip install tqdm")
        return False
    
    return True


def test_data_file():
    """Test if transcript file exists"""
    print("\n🔍 Testing data file...")
    
    try:
        with open('transcripts_konkani_cleaned.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  ✅ Found {len(data)} transcripts")
        
        # Show sample
        if len(data) > 0:
            sample = data[0]
            print(f"\n  📝 Sample transcript:")
            print(f"     {sample['transcript'][:100]}...")
        
        return True
    except FileNotFoundError:
        print("  ❌ transcripts_konkani_cleaned.json not found")
        return False
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False


def test_auto_labeling():
    """Test auto-labeling on a few samples"""
    print("\n🔍 Testing auto-labeling (this may take a minute)...")
    
    try:
        from scripts.auto_label_ner import NERAutoLabeler
        
        # Create labeler
        labeler = NERAutoLabeler()
        
        # Test on a simple text
        test_text = "मी मुंबईंत गूगलांत काम करतां"
        result = labeler.label_text(test_text)
        
        if result:
            print(f"\n  ✅ Auto-labeling works!")
            print(f"\n  📝 Test example:")
            print(f"     Text: {test_text}")
            print(f"     Tokens: {result['tokens']}")
            print(f"     Labels: {result['labels']}")
            
            # Check if any entities found
            entities_found = any(label != 'O' for label in result['labels'])
            if entities_found:
                print(f"\n  🎯 Found entities:")
                for token, label in zip(result['tokens'], result['labels']):
                    if label != 'O':
                        print(f"     {token} → {label}")
            else:
                print(f"\n  ℹ️  No entities found in test text (this is okay)")
            
            return True
        else:
            print("  ❌ Auto-labeling failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test if NER model can be created"""
    print("\n🔍 Testing model creation...")
    
    try:
        from models.konkani_ner import create_ner_model
        
        # Try with CRF
        try:
            model = create_ner_model(
                vocab_size=1000,
                num_tags=9,
                use_crf=True,
                char_vocab_size=200
            )
            print("  ✅ BiLSTM-CRF model created")
        except:
            # Try without CRF
            model = create_ner_model(
                vocab_size=1000,
                num_tags=9,
                use_crf=False
            )
            print("  ✅ BiLSTM model created (without CRF)")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"     Parameters: {num_params:,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("KONKANI NER SETUP TEST")
    print("="*70)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Data file
    results.append(("Data file", test_data_file()))
    
    # Test 3: Auto-labeling (only if imports work)
    if results[0][1]:
        results.append(("Auto-labeling", test_auto_labeling()))
    
    # Test 4: Model creation
    if results[0][1]:
        results.append(("Model creation", test_model_creation()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou're ready to start NER training:")
        print("  1. python3 scripts/auto_label_ner.py")
        print("  2. python3 train_konkani_ner.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
