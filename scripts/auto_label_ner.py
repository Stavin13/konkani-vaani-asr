"""
Auto-label Konkani transcripts with NER tags using pre-trained model
Then prepare data for training custom NER model
"""

import json
import torch
from transformers import pipeline
from tqdm import tqdm
import re

class NERAutoLabeler:
    """Auto-label Konkani text with NER tags"""
    
    def __init__(self):
        print("📦 Loading pre-trained multilingual NER model...")
        # Using XLM-RoBERTa trained on multiple languages
        self.ner_pipeline = pipeline(
            "ner",
            model="Davlan/xlm-roberta-base-ner-hrl",
            aggregation_strategy="simple"
        )
        print("✅ Model loaded!\n")
        
        # BIO tag mapping
        self.tag_map = {
            'PER': ['B-PER', 'I-PER'],
            'ORG': ['B-ORG', 'I-ORG'],
            'LOC': ['B-LOC', 'I-LOC'],
            'MISC': ['B-MISC', 'I-MISC']
        }
        
        # Create label to ID mapping
        self.label2id = {'O': 0}
        idx = 1
        for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
            self.label2id[f'B-{entity_type}'] = idx
            idx += 1
            self.label2id[f'I-{entity_type}'] = idx
            idx += 1
        
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def tokenize_text(self, text):
        """Simple word tokenization for Devanagari"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        return tokens
    
    def align_labels(self, text, entities):
        """Convert entity spans to BIO tags for each token"""
        tokens = self.tokenize_text(text)
        labels = ['O'] * len(tokens)
        
        # Track character positions
        char_to_token = {}
        current_pos = 0
        for i, token in enumerate(tokens):
            token_start = text.find(token, current_pos)
            if token_start != -1:
                for j in range(token_start, token_start + len(token)):
                    char_to_token[j] = i
                current_pos = token_start + len(token)
        
        # Assign BIO tags based on entities
        for entity in entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['entity_group']
            
            # Find tokens that overlap with this entity
            token_indices = set()
            for char_pos in range(start, end):
                if char_pos in char_to_token:
                    token_indices.add(char_to_token[char_pos])
            
            # Assign B- and I- tags
            token_indices = sorted(token_indices)
            for idx, token_idx in enumerate(token_indices):
                if idx == 0:
                    labels[token_idx] = f'B-{entity_type}'
                else:
                    labels[token_idx] = f'I-{entity_type}'
        
        return tokens, labels
    
    def label_text(self, text):
        """Generate NER labels for a single text"""
        if not text or len(text.strip()) < 3:
            return None
        
        try:
            # Get entities from pre-trained model
            entities = self.ner_pipeline(text)
            
            # Convert to BIO format
            tokens, labels = self.align_labels(text, entities)
            
            return {
                'text': text,
                'tokens': tokens,
                'labels': labels,
                'label_ids': [self.label2id.get(label, 0) for label in labels]
            }
        except Exception as e:
            print(f"⚠️  Error labeling text: {e}")
            return None
    
    def label_dataset(self, input_file, output_file, max_samples=None):
        """Label entire dataset"""
        print(f"📂 Loading data from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"📊 Processing {len(data)} samples...\n")
        
        labeled_data = []
        skipped = 0
        
        for item in tqdm(data, desc="Auto-labeling"):
            transcript = item.get('transcript', '')
            
            result = self.label_text(transcript)
            if result:
                result['segment_id'] = item.get('segment_id')
                result['audio_filepath'] = item.get('audio_filepath')
                labeled_data.append(result)
            else:
                skipped += 1
        
        print(f"\n✅ Labeled {len(labeled_data)} samples")
        print(f"⚠️  Skipped {skipped} samples (too short or errors)")
        
        # Save labeled data
        print(f"\n💾 Saving to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, ensure_ascii=False, indent=2)
        
        # Save label mapping
        label_map_file = output_file.replace('.json', '_label_map.json')
        with open(label_map_file, 'w', encoding='utf-8') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved label mapping to: {label_map_file}")
        
        # Print statistics
        self.print_statistics(labeled_data)
        
        return labeled_data
    
    def print_statistics(self, labeled_data):
        """Print dataset statistics"""
        print("\n" + "="*70)
        print("📊 DATASET STATISTICS")
        print("="*70)
        
        total_tokens = 0
        entity_counts = {label: 0 for label in self.label2id.keys()}
        
        for item in labeled_data:
            total_tokens += len(item['tokens'])
            for label in item['labels']:
                entity_counts[label] += 1
        
        print(f"\nTotal samples: {len(labeled_data)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Avg tokens per sample: {total_tokens / len(labeled_data):.1f}")
        
        print("\n📋 Label distribution:")
        for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            percentage = (count / total_tokens) * 100
            print(f"  {label:10s}: {count:6,} ({percentage:5.2f}%)")
        
        print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-label NER data')
    parser.add_argument('--input', type=str, 
                        default='transcripts_konkani_cleaned.json',
                        help='Input transcript file')
    parser.add_argument('--output', type=str,
                        default='data/ner_labeled_data.json',
                        help='Output labeled data file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Create labeler
    labeler = NERAutoLabeler()
    
    # Label dataset
    labeled_data = labeler.label_dataset(
        args.input,
        args.output,
        max_samples=args.max_samples
    )
    
    print("\n✅ Auto-labeling complete!")
    print(f"   Labeled data saved to: {args.output}")
    print(f"   Ready for training custom NER model!")


if __name__ == "__main__":
    main()
