"""
Auto-translate Konkani transcripts to English using pre-trained model
Then prepare data for training custom translation model
"""

import json
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

class TranslationAutoLabeler:
    """Auto-translate Konkani to English"""
    
    def __init__(self):
        print("📦 Loading pre-trained translation model (M2M-100)...")
        print("   This model supports 100 languages including Indic languages\n")
        
        # Using M2M-100 (Many-to-Many 100 languages)
        model_name = "facebook/m2m100_418M"
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}!\n")
    
    def translate_text(self, text, src_lang="hi", tgt_lang="en"):
        """
        Translate text from source to target language
        
        Args:
            text: Input text
            src_lang: Source language code (hi for Hindi/Devanagari)
            tgt_lang: Target language code (en for English)
        """
        if not text or len(text.strip()) < 2:
            return None
        
        try:
            # Set source language
            self.tokenizer.src_lang = src_lang
            
            # Encode
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate translation
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            return translation
            
        except Exception as e:
            print(f"⚠️  Error translating: {e}")
            return None
    
    def translate_dataset(self, input_file, output_file, max_samples=None):
        """Translate entire dataset"""
        print(f"📂 Loading data from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"📊 Processing {len(data)} samples...\n")
        
        translated_data = []
        skipped = 0
        
        for item in tqdm(data, desc="Translating"):
            konkani_text = item.get('transcript', '')
            
            translation = self.translate_text(konkani_text, src_lang="hi", tgt_lang="en")
            
            if translation:
                translated_data.append({
                    'segment_id': item.get('segment_id'),
                    'audio_filepath': item.get('audio_filepath'),
                    'konkani': konkani_text,
                    'english': translation
                })
            else:
                skipped += 1
        
        print(f"\n✅ Translated {len(translated_data)} samples")
        print(f"⚠️  Skipped {skipped} samples (too short or errors)")
        
        # Save translated data
        print(f"\n💾 Saving to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved translation data")
        
        # Print sample translations
        self.print_samples(translated_data[:5])
        
        return translated_data
    
    def print_samples(self, samples):
        """Print sample translations"""
        print("\n" + "="*70)
        print("📝 SAMPLE TRANSLATIONS")
        print("="*70)
        
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Konkani: {sample['konkani'][:100]}...")
            print(f"  English: {sample['english'][:100]}...")
        
        print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-translate Konkani to English')
    parser.add_argument('--input', type=str, 
                        default='transcripts_konkani_cleaned.json',
                        help='Input transcript file')
    parser.add_argument('--output', type=str,
                        default='data/translation_data.json',
                        help='Output translation data file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Create translator
    translator = TranslationAutoLabeler()
    
    # Translate dataset
    translated_data = translator.translate_dataset(
        args.input,
        args.output,
        max_samples=args.max_samples
    )
    
    print("\n✅ Auto-translation complete!")
    print(f"   Translation data saved to: {args.output}")
    print(f"   Ready for training custom translation model!")


if __name__ == "__main__":
    main()
