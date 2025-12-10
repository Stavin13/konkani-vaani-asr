#!/usr/bin/env python3
"""
Quick demo of the KonkaniVani pipeline
"""
from pipeline import KonkaniPipeline
import sys


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_results(results):
    """Pretty print results"""
    print("\n📝 KONKANI TEXT:")
    print(f"   {results['konkani_text']}")
    
    if 'english_text' in results:
        print("\n🌐 ENGLISH TRANSLATION:")
        print(f"   {results['english_text']}")
    
    if 'emotion' in results:
        emotion = results['emotion']
        print("\n😊 EMOTION:")
        print(f"   {emotion['label'].upper()} ({emotion['confidence']:.1%} confidence)")
        print("\n   All scores:")
        for emo, score in sorted(emotion['all_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"   {emo:10s} {bar:20s} {score:.1%}")
    
    if 'entities' in results:
        print("\n🏷️  NAMED ENTITIES:")
        if results['entities']:
            for entity_text, entity_type, start, end in results['entities']:
                print(f"   {entity_type:8s} → {entity_text}")
        else:
            print("   (none detected)")


def demo_text_processing():
    """Demo text processing"""
    print_header("TEXT PROCESSING DEMO")
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    pipeline = KonkaniPipeline()
    
    # Test cases
    test_texts = [
        "हांव घरा वचता",
        "हांव खुश आसा",
        "तूं कसो आसा",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n\n{'─'*70}")
        print(f"Example {i}/{len(test_texts)}")
        print('─'*70)
        
        results = pipeline.process_text(
            text,
            include_translation=True,
            include_emotion=True,
            include_ner=True
        )
        
        print_results(results)


def demo_individual_models():
    """Demo individual model usage"""
    print_header("INDIVIDUAL MODELS DEMO")
    
    # Translation
    print("\n1️⃣  TRANSLATION MODEL")
    from models import TranslationModel
    
    translator = TranslationModel()
    
    konkani = "हांव घरा वचता"
    english = translator.konkani_to_english(konkani)
    print(f"   Konkani: {konkani}")
    print(f"   English: {english}")
    
    english = "I am happy"
    konkani = translator.english_to_konkani(english)
    print(f"\n   English: {english}")
    print(f"   Konkani: {konkani}")
    
    # Emotion
    print("\n2️⃣  EMOTION MODEL")
    from models import EmotionModel
    
    emotion_model = EmotionModel()
    
    texts = [
        ("हांव खुश आसा", "happy"),
        ("हांव दुःखी आसा", "sad"),
        ("हांव रागीत आसा", "angry"),
    ]
    
    for text, expected in texts:
        emotion, confidence, _ = emotion_model.predict(text)
        print(f"   {text:20s} → {emotion:10s} ({confidence:.1%})")
    
    # NER
    print("\n3️⃣  NER MODEL")
    from models import NERModel
    
    ner = NERModel()
    
    text = "हांव Mumbai वचता"
    entities = ner.predict(text)
    print(f"   Text: {text}")
    if entities:
        for entity_text, entity_type, start, end in entities:
            print(f"   Found: {entity_type} = {entity_text}")
    else:
        print("   No entities detected")


def interactive_mode():
    """Interactive demo mode"""
    print_header("INTERACTIVE MODE")
    
    print("\n🚀 Initializing pipeline...")
    pipeline = KonkaniPipeline()
    
    print("\n✅ Ready! Enter Konkani text (or 'quit' to exit)")
    print("   Example: हांव घरा वचता")
    
    while True:
        print("\n" + "─"*70)
        text = input("\n📝 Konkani text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Bye!")
            break
        
        if not text:
            continue
        
        try:
            results = pipeline.process_text(
                text,
                include_translation=True,
                include_emotion=True,
                include_ner=True
            )
            print_results(results)
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KonkaniVani Pipeline Demo')
    parser.add_argument('--mode', type=str, default='text',
                       choices=['text', 'models', 'interactive'],
                       help='Demo mode')
    args = parser.parse_args()
    
    print("="*70)
    print("  🎤 KONKANIVANI PIPELINE DEMO")
    print("="*70)
    
    try:
        if args.mode == 'text':
            demo_text_processing()
        elif args.mode == 'models':
            demo_individual_models()
        elif args.mode == 'interactive':
            interactive_mode()
        
        print("\n" + "="*70)
        print("  ✅ DEMO COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  • Run web app: streamlit run app.py")
        print("  • Read guide: cat USAGE_GUIDE.md")
        print("  • Use in code: from pipeline import KonkaniPipeline")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Bye!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: python test_pipeline.py")
        print("  2. Check checkpoint paths")
        print("  3. Verify dependencies installed")


if __name__ == '__main__':
    main()
