#!/usr/bin/env python3
"""
Simple text-only pipeline test for Konkani NLP
"""
import sys
from pathlib import Path

# Add deployment directory to path
sys.path.insert(0, str(Path(__file__).parent / "deployment"))

def test_text_processing(text):
    """Test text processing without ASR"""
    print(f"🔤 Input text: {text}")
    
    # For now, let's just test basic functionality
    try:
        # Test translation (if available)
        print("🔄 Translation: [Would translate Konkani to English]")
        print("📝 NER: [Would extract named entities]") 
        print("😊 Emotion: [Would detect emotion]")
        
        # Mock results for demonstration
        results = {
            "input_text": text,
            "translation": "I am going home",  # Mock translation
            "entities": [],  # Mock NER results
            "emotion": "neutral",  # Mock emotion
            "confidence": 0.85
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Error in text processing: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Konkani text processing")
    parser.add_argument("--text", required=True, help="Konkani text to process")
    
    args = parser.parse_args()
    
    print("🚀 Testing Konkani Text Pipeline...")
    results = test_text_processing(args.text)
    
    if results:
        print("\n✅ Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Processing failed")