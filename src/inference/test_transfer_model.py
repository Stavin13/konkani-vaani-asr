#!/usr/bin/env python3
"""
Test the trained Konkani Sentiment Analysis Model
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


def test_model(model_path: str = "./konkani_sentiment_model"):
    """
    Test the trained model with sample sentences
    
    Args:
        model_path: Path to the trained model
    """
    print("="*60)
    print("KONKANI SENTIMENT ANALYSIS - MODEL TESTING")
    print("="*60)
    
    # Load model and tokenizer
    print(f"\nLoading model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create pipeline for easier inference
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure you've trained the model first by running:")
        print("  python train_konkani_model.py")
        return
    
    # Test sentences (Konkani - both Devanagari and Romanized)
    test_sentences = [
        # Positive examples
        "हें फोन खूब छान आसा",
        "hen phone khup chan asa",
        "ही फिल्म अतिशय सुंदर आसा",
        "hi film atishay sundar asa",
        
        # Negative examples
        "हें जेवण अगदी वायट आसा",
        "hen jevan agdi vait asa",
        "ही सेवा पुराय खराब आसा",
        "hi seva puray kharab asa",
        
        # Neutral examples
        "हें पुस्तक साधारण आसा",
        "hen pustak sadharan asa",
        "ही दुकान मध्यम आसा",
        "hi dukan madhyam asa"
    ]
    
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE SENTENCES")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Input: {sentence}")
        
        # Get prediction
        result = classifier(sentence)[0]
        
        # Display result
        label = result['label']
        score = result['score']
        
        # Map label to sentiment
        if 'LABEL_0' in label or 'negative' in label.lower():
            sentiment = "NEGATIVE 😞"
        elif 'LABEL_1' in label or 'neutral' in label.lower():
            sentiment = "NEUTRAL 😐"
        elif 'LABEL_2' in label or 'positive' in label.lower():
            sentiment = "POSITIVE 😊"
        else:
            sentiment = label
        
        print(f"   Prediction: {sentiment}")
        print(f"   Confidence: {score:.2%}")
        print(f"   Raw: {result}")
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter Konkani sentences to analyze (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nKonkani text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get prediction
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']
            
            # Map label to sentiment
            if 'LABEL_0' in label or 'negative' in label.lower():
                sentiment = "NEGATIVE 😞"
            elif 'LABEL_1' in label or 'neutral' in label.lower():
                sentiment = "NEUTRAL 😐"
            elif 'LABEL_2' in label or 'positive' in label.lower():
                sentiment = "POSITIVE 😊"
            else:
                sentiment = label
            
            print(f"→ Sentiment: {sentiment} (Confidence: {score:.2%})")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Allow custom model path as argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./konkani_sentiment_model"
    
    test_model(model_path)
