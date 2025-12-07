"""
Demo: ASR + NER Pipeline
Transcribe audio and extract named entities
"""

import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_ner import KonkaniNER


class ASRNERPipeline:
    """Combined ASR + NER pipeline"""
    
    def __init__(self, ner_checkpoint_path, vocab_path, label_map_path):
        print("📦 Loading NER model...")
        
        # Load vocabularies
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocabs = json.load(f)
        
        self.word2id = vocabs['word2id']
        self.char2id = vocabs['char2id']
        self.id2word = {v: k for k, v in self.word2id.items()}
        
        # Load label mapping
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        self.label2id = label_data['label2id']
        self.id2label = label_data['id2label']
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = KonkaniNER(
            vocab_size=len(self.word2id),
            char_vocab_size=len(self.char2id),
            num_tags=len(self.label2id),
            embedding_dim=128,
            char_embedding_dim=32,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        )
        
        # Load checkpoint
        checkpoint = torch.load(ner_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ NER model loaded!\n")
    
    def tokenize(self, text):
        """Simple tokenization"""
        import re
        return re.findall(r'\S+', text)
    
    def encode_tokens(self, tokens, max_char_len=20):
        """Convert tokens to IDs"""
        word_ids = []
        char_ids = []
        
        for token in tokens:
            # Word ID
            word_id = self.word2id.get(token, self.word2id.get('<UNK>', 1))
            word_ids.append(word_id)
            
            # Character IDs
            token_char_ids = []
            for char in token[:max_char_len]:
                char_id = self.char2id.get(char, self.char2id.get('<UNK>', 1))
                token_char_ids.append(char_id)
            
            # Pad to max_char_len
            while len(token_char_ids) < max_char_len:
                token_char_ids.append(0)
            
            char_ids.append(token_char_ids)
        
        return word_ids, char_ids
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        # Tokenize
        tokens = self.tokenize(text)
        if not tokens:
            return []
        
        # Encode
        word_ids, char_ids = self.encode_tokens(tokens)
        
        # Convert to tensors
        word_ids_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        char_ids_tensor = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(word_ids_tensor, char_ids_tensor)
        
        # Decode predictions
        pred_tags = predictions[0]
        pred_labels = [self.id2label[str(tag_id)] for tag_id in pred_tags]
        
        # Extract entities
        entities = []
        current_entity = None
        entity_tokens = []
        entity_start = None
        
        for idx, (token, label) in enumerate(zip(tokens, pred_labels)):
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': current_entity,
                        'start': entity_start,
                        'end': idx - 1
                    })
                
                # Start new entity
                current_entity = label[2:]
                entity_tokens = [token]
                entity_start = idx
            
            elif label.startswith('I-') and current_entity:
                # Continue entity
                entity_tokens.append(token)
            
            else:
                # Not an entity
                if current_entity:
                    entities.append({
                        'text': ' '.join(entity_tokens),
                        'type': current_entity,
                        'start': entity_start,
                        'end': idx - 1
                    })
                    current_entity = None
                    entity_tokens = []
        
        # Save last entity
        if current_entity:
            entities.append({
                'text': ' '.join(entity_tokens),
                'type': current_entity,
                'start': entity_start,
                'end': len(tokens) - 1
            })
        
        return entities
    
    def process_transcript(self, transcript):
        """Process ASR transcript and extract entities"""
        print(f"📝 Transcript: {transcript}\n")
        
        entities = self.extract_entities(transcript)
        
        if entities:
            print("🏷️  Named Entities:")
            print("-" * 60)
            for entity in entities:
                print(f"  [{entity['type']}] {entity['text']}")
            print("-" * 60)
        else:
            print("ℹ️  No entities found")
        
        return entities


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASR + NER Pipeline Demo')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/ner/best_ner_model.pt',
                        help='Path to NER checkpoint')
    parser.add_argument('--vocab', type=str,
                        default='checkpoints/ner/vocabularies.json',
                        help='Path to vocabularies')
    parser.add_argument('--labels', type=str,
                        default='data/ner_labeled_data_label_map.json',
                        help='Path to label mapping')
    parser.add_argument('--transcript', type=str,
                        help='Transcript to process')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ASRNERPipeline(args.checkpoint, args.vocab, args.labels)
    
    print("="*70)
    print("🎤 ASR + NER PIPELINE DEMO")
    print("="*70 + "\n")
    
    # Test transcripts
    test_transcripts = [
        "मझ्या वाताराण न्योजिक स्तान खाय असा माझा वाारां कलंगुट चरच असा",
        "ना आलली कित्या ओडखण आलली आणि टi लबत शकोन लकांबरा मिसोळजपा",
        "ते आवआटा आणि जेन्ना आणि ॉ कॉllेज कितीसोलतू का लाक"
    ]
    
    if args.transcript:
        test_transcripts = [args.transcript]
    
    for i, transcript in enumerate(test_transcripts, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}")
        print('='*70 + "\n")
        
        entities = pipeline.process_transcript(transcript)
        
        # Show structured output
        if entities:
            print("\n📊 Structured Output:")
            print(json.dumps(entities, ensure_ascii=False, indent=2))
    
    print("\n" + "="*70)
    print("✅ Pipeline demo complete!")
    print("="*70)
    
    print("\n💡 Usage Tips:")
    print("  1. Integrate with ASR model for end-to-end processing")
    print("  2. Use entities for downstream tasks (search, analytics)")
    print("  3. Fine-tune on domain-specific data for better accuracy")


if __name__ == "__main__":
    main()
