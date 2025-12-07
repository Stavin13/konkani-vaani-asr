#!/usr/bin/env python3
"""
Predict sentiment for Konkani text using trained model.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from konkani.inference.sentiment_predictor import SentimentPredictor
from config.paths import Paths


def main():
    """Main prediction script"""
    
    # Load model
    model_dir = Paths.MODELS_SENTIMENT / "custom_konkani_model"
    
    if not model_dir.exists():
        print(f"Error: Model not found at {model_dir}")
        print("Please train a model first using: python scripts/sentiment/train_custom.py")
        return
    
    print("Loading model...")
    predictor = SentimentPredictor(model_dir)
    print("✓ Model loaded successfully\n")
    
    # Interactive prediction
    print("Konkani Sentiment Analysis")
    print("="*60)
    print("Enter Konkani text to analyze (or 'quit' to exit)")
    print("="*60)
    
    while True:
        text = input("\nText: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        # Predict
        result = predictor.predict(text)
        
        # Display results
        print(f"\nSentiment: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")


if __name__ == "__main__":
    main()
