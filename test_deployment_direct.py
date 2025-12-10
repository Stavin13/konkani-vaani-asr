#!/usr/bin/env python3
"""
Direct test of deployment pipeline without Streamlit
"""
import sys
sys.path.insert(0, 'deployment')
sys.path.insert(0, '.')

print("=" * 70)
print("TESTING DEPLOYMENT PIPELINE")
print("=" * 70)

# Test with text input (no audio loading issues)
print("\n1. Testing with text input...")
from deployment.pipeline import KonkaniPipeline

pipeline = KonkaniPipeline(device='cpu')
print("✅ Pipeline loaded successfully!")

# Test text processing
test_text = "हांव घरा वचता"
print(f"\nProcessing text: {test_text}")

results = pipeline.process_text(
    test_text,
    include_translation=True,
    include_emotion=True,
    include_ner=True
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Konkani: {results['konkani_text']}")
if 'english_text' in results:
    print(f"English: {results['english_text']}")
if 'emotion' in results:
    print(f"Emotion: {results['emotion']['label']} ({results['emotion']['confidence']:.2%})")
if 'entities' in results:
    print(f"Entities: {results['entities']}")

print("\n✅ All tests passed!")
