"""
Test the trained Konkani NER model
"""

import torch
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_ner import KonkaniNER


class NERPredictor:
    """Load and use trained NER model for predictions"""
    
    def __init__(self, checkpoint_path, vocab_path, label_map_path):
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from: {checkpoint_path}")
        print(f"   Device: {self.device}")
        print(f"   Vocab size: {len(self.word2id)}")
        print(f"   Char vocab size: {len(self.char2id)}")
        print(f"   Num tags: {len(self.label2id)}\n")
    
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
    
    def predict(self, text):
        """Predict NER tags for text"""
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
        pred_tags = predictions[0]  # First (and only) sequence
        pred_labels = [self.id2label[str(tag_id)] for tag_id in pred_tags]
        
        # Combine tokens and labels
        results = []
        for token, label in zip(tokens, pred_labels):
            results.append({
                'token': token,
                'label': label
            })
        
        return results
    
    def print_predictions(self, text, predictions):
        """Pretty print predictions"""
        print(f"Text: {text}\n")
        print("Predictions:")
        print("-" * 60)
        
        current_entity = None
        entity_tokens = []
        
        for pred in predictions:
            token = pred['token']
            label = pred['label']
            
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entity_text = ' '.join(entity_tokens)
                    print(f"  [{current_entity}] {entity_text}")
                
                current_entity = label[2:]  # Remove 'B-'
                entity_tokens = [token]
            
            elif label.startswith('I-') and current_entity:
                # Continuation of entity
                entity_tokens.append(token)
            
            else:
                # Not an entity
                if current_entity:
                    entity_text = ' '.join(entity_tokens)
                    print(f"  [{current_entity}] {entity_text}")
                    current_entity = None
                    entity_tokens = []
        
        # Print last entity if exists
        if current_entity:
            entity_text = ' '.join(entity_tokens)
            print(f"  [{current_entity}] {entity_text}")
        
        print("-" * 60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NER model')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/ner/best_ner_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str,
                        default='checkpoints/ner/vocabularies.json',
                        help='Path to vocabularies')
    parser.add_argument('--labels', type=str,
                        default='data/ner_labeled_data_label_map.json',
                        help='Path to label mapping')
    parser.add_argument('--text', type=str,
                        help='Text to predict (optional)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NERPredictor(args.checkpoint, args.vocab, args.labels)
    
    # Test examples
    test_texts = [
        "देव बर दिसनयकत कालेजाल आंगा मझो वांगडा हाजीर असात",
        "मुन मका खूप खुशी पहलो विचार हा की आतोा जेवरांत शिकपाेला",
        "साउथ सोडून लाक तुझो अकॉ डेली जीवित सावतांत"
    ]
    
    if args.text:
        test_texts = [args.text]
    
    print("="*70)
    print("🧪 TESTING NER MODEL")
    print("="*70 + "\n")
    
    for text in test_texts:
        predictions = predictor.predict(text)
        predictor.print_predictions(text, predictions)
    
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()
