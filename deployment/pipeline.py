"""
Complete Konkani NLP Pipeline
Integrates ASR, Translation, NER, and Emotion Detection
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path to access main models folder
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from local models package (deployment/models/)
from models.asr_model import ASRModel
from models.translation_model import TranslationModel
from models.emotion_model import EmotionModel
from models.ner_model import NERModel


class KonkaniPipeline:
    """Complete pipeline for Konkani audio/text processing"""
    
    def __init__(self, device=None):
        """Initialize all models"""
        print("🚀 Initializing Konkani NLP Pipeline...")
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        print(f"📱 Device: {device}")
        
        # Initialize models
        print("\n1️⃣  Loading ASR model...")
        self.asr = ASRModel(device=device)
        
        print("\n2️⃣  Loading Translation model...")
        # Try finetuned first, fallback to base NLLB
        finetuned_path = "../checkpoints/nllb_finetuned/final"
        self.translator = TranslationModel(
            model_path=finetuned_path if Path(finetuned_path).exists() else None,
            device=device
        )
        
        print("\n3️⃣  Loading Emotion model...")
        self.emotion = EmotionModel(device=device)
        
        print("\n4️⃣  Loading NER model...")
        self.ner = NERModel(device=device)
        
        print("\n✅ Pipeline ready!\n")
    
    def process_audio(self, audio_path, include_translation=True, 
                     include_emotion=True, include_ner=True):
        """
        Complete pipeline: Audio → Transcription → Translation → Analysis
        
        Args:
            audio_path: Path to audio file
            include_translation: Whether to translate to English
            include_emotion: Whether to detect emotion
            include_ner: Whether to extract entities
        
        Returns:
            results: Dictionary with all results
        """
        results = {}
        
        # Step 1: Transcribe audio
        print("🎤 Transcribing audio...")
        konkani_text = self.asr.transcribe(audio_path)
        results['konkani_text'] = konkani_text
        print(f"   Konkani: {konkani_text}")
        
        # Step 2: Translate
        if include_translation:
            print("\n🌐 Translating to English...")
            english_text = self.translator.konkani_to_english(konkani_text)
            results['english_text'] = english_text
            print(f"   English: {english_text}")
        
        # Step 3: Emotion detection
        if include_emotion:
            print("\n😊 Detecting emotion...")
            emotion, confidence, all_scores = self.emotion.predict(konkani_text)
            results['emotion'] = {
                'label': emotion,
                'confidence': confidence,
                'all_scores': all_scores
            }
            print(f"   Emotion: {emotion} ({confidence:.2%})")
        
        # Step 4: Named Entity Recognition
        if include_ner:
            print("\n🏷️  Extracting entities...")
            entities = self.ner.predict(konkani_text)
            results['entities'] = entities
            if entities:
                for entity_text, entity_type, start, end in entities:
                    print(f"   {entity_type}: {entity_text}")
            else:
                print("   No entities found")
        
        return results
    
    def process_text(self, konkani_text, include_translation=True,
                    include_emotion=True, include_ner=True):
        """
        Process Konkani text (skip ASR)
        
        Args:
            konkani_text: Konkani text input
            include_translation: Whether to translate
            include_emotion: Whether to detect emotion
            include_ner: Whether to extract entities
        
        Returns:
            results: Dictionary with all results
        """
        results = {'konkani_text': konkani_text}
        
        # Translation
        if include_translation:
            print("🌐 Translating to English...")
            english_text = self.translator.konkani_to_english(konkani_text)
            results['english_text'] = english_text
            print(f"   English: {english_text}")
        
        # Emotion
        if include_emotion:
            print("\n😊 Detecting emotion...")
            emotion, confidence, all_scores = self.emotion.predict(konkani_text)
            results['emotion'] = {
                'label': emotion,
                'confidence': confidence,
                'all_scores': all_scores
            }
            print(f"   Emotion: {emotion} ({confidence:.2%})")
        
        # NER
        if include_ner:
            print("\n🏷️  Extracting entities...")
            entities = self.ner.predict(konkani_text)
            results['entities'] = entities
            if entities:
                for entity_text, entity_type, start, end in entities:
                    print(f"   {entity_type}: {entity_text}")
            else:
                print("   No entities found")
        
        return results


def main():
    """Test the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Konkani NLP Pipeline')
    parser.add_argument('--audio', type=str, help='Audio file path')
    parser.add_argument('--text', type=str, help='Konkani text')
    parser.add_argument('--device', type=str, default=None, choices=['mps', 'cuda', 'cpu'])
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = KonkaniPipeline(device=args.device)
    
    if args.audio:
        # Process audio
        results = pipeline.process_audio(args.audio)
    elif args.text:
        # Process text
        results = pipeline.process_text(args.text)
    else:
        # Demo mode
        print("Demo mode - provide --audio or --text for real processing")
        demo_text = "हांव घरा वचता"
        results = pipeline.process_text(demo_text)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(results)


if __name__ == '__main__':
    main()
