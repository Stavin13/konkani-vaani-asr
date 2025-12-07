"""
Simple demo of NER model with sample data
"""

import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_ner import KonkaniNER


def load_sample_data():
    """Load a few samples from the labeled data"""
    data_path = 'data/ner_labeled_data.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find samples with entities
    samples_with_entities = []
    for item in data:
        if any(label != 'O' for label in item['labels']):
            samples_with_entities.append(item)
            if len(samples_with_entities) >= 5:
                break
    
    return samples_with_entities


def main():
    print("📦 Loading NER model and data...\n")
    
    # Load vocabularies
    with open('checkpoints/ner/vocabularies.json', 'r', encoding='utf-8') as f:
        vocabs = json.load(f)
    
    word2id = vocabs['word2id']
    char2id = vocabs['char2id']
    
    # Load label mapping
    with open('data/ner_labeled_data_label_map.json', 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    id2label = label_data['id2label']
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = KonkaniNER(
        vocab_size=len(word2id),
        char_vocab_size=len(char2id),
        num_tags=len(id2label),
        embedding_dim=128,
        char_embedding_dim=32,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    checkpoint = torch.load('checkpoints/ner/best_ner_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded!")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    best_f1 = checkpoint.get('best_f1', 0)
    if isinstance(best_f1, (int, float)):
        print(f"   Best F1: {best_f1:.4f}\n")
    else:
        print(f"   Best F1: {best_f1}\n")
    
    # Load sample data
    samples = load_sample_data()
    
    print("="*70)
    print("🧪 TESTING WITH LABELED DATA SAMPLES")
    print("="*70 + "\n")
    
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(f"Text: {sample['text']}\n")
        
        print("Ground Truth Entities:")
        print("-" * 60)
        
        current_entity = None
        entity_tokens = []
        
        for token, label in zip(sample['tokens'], sample['labels']):
            if label.startswith('B-'):
                if current_entity:
                    entity_text = ' '.join(entity_tokens)
                    print(f"  [{current_entity}] {entity_text}")
                
                current_entity = label[2:]
                entity_tokens = [token]
            
            elif label.startswith('I-') and current_entity:
                entity_tokens.append(token)
            
            else:
                if current_entity:
                    entity_text = ' '.join(entity_tokens)
                    print(f"  [{current_entity}] {entity_text}")
                    current_entity = None
                    entity_tokens = []
        
        if current_entity:
            entity_text = ' '.join(entity_tokens)
            print(f"  [{current_entity}] {entity_text}")
        
        print("-" * 60 + "\n")
    
    print("\n✅ Demo complete!")
    print(f"\nModel Statistics:")
    print(f"  Total vocabulary: {len(word2id):,} words")
    print(f"  Character vocabulary: {len(char2id)} chars")
    print(f"  Entity types: PER, ORG, LOC, MISC")


if __name__ == "__main__":
    main()
