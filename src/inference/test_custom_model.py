#!/usr/bin/env python3
"""
Test Custom Konkani Sentiment Model
"""

import torch
import torch.nn as nn
import pickle
import sys


class CustomKonkaniSentimentModel(nn.Module):
    """Custom Neural Network for Konkani Sentiment Analysis"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(CustomKonkaniSentimentModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def attention_layer(self, lstm_output, lengths):
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        return attended, attention_weights
    
    def forward(self, input_ids, lengths=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        attended, attention_weights = self.attention_layer(lstm_out, lengths)
        x = self.relu(self.fc1(attended))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits, attention_weights


class KonkaniSentimentPredictor:
    """Predictor for custom Konkani sentiment model"""
    
    def __init__(self, model_dir: str = "../../models/custom_konkani_model"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        with open(f"{model_dir}/tokenizer.pkl", 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.word2idx = tokenizer_data['word2idx']
        self.idx2word = tokenizer_data['idx2word']
        
        # Load model
        checkpoint = torch.load(f"{model_dir}/best_model.pt", map_location=self.device)
        config = checkpoint['model_config']
        
        self.model = CustomKonkaniSentimentModel(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.label_names = ['negative', 'neutral', 'positive']
        self.label_emojis = ['😞', '😐', '😊']
    
    def tokenize(self, text: str) -> list:
        """Tokenize text"""
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def encode(self, text: str, max_length: int = 128) -> torch.Tensor:
        """Encode text to tensor"""
        words = self.tokenize(text)
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        return torch.tensor([indices], dtype=torch.long)
    
    def predict(self, text: str):
        """Predict sentiment for text"""
        # Encode
        input_ids = self.encode(text).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attention_weights = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'label': self.label_names[predicted_class],
            'emoji': self.label_emojis[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            }
        }


def test_model(model_dir: str = "../../models/custom_konkani_model"):
    """Test the custom model"""
    
    print("="*60)
    print("CUSTOM KONKANI SENTIMENT ANALYSIS - MODEL TESTING")
    print("="*60)
    
    try:
        predictor = KonkaniSentimentPredictor(model_dir)
        print("✓ Custom model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure you've trained the model first:")
        print("  python train_custom_model.py")
        return
    
    # Test sentences
    test_sentences = [
        # Positive
        "हें फोन खूब छान आसा",
        "hen phone khup chan asa",
        "ही फिल्म अतिशय सुंदर आसा",
        "hi film atishay sundar asa",
        
        # Negative
        "हें जेवण अगदी वायट आसा",
        "hen jevan agdi vait asa",
        "ही सेवा पुराय खराब आसा",
        "hi seva puray kharab asa",
        
        # Neutral
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
        
        result = predictor.predict(sentence)
        
        print(f"   Sentiment: {result['label'].upper()} {result['emoji']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"     {label}: {prob:.2%}")
    
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
            
            result = predictor.predict(user_input)
            
            print(f"→ Sentiment: {result['label'].upper()} {result['emoji']}")
            print(f"→ Confidence: {result['confidence']:.2%}")
            print(f"→ Probabilities: Neg={result['probabilities']['negative']:.1%} | "
                  f"Neu={result['probabilities']['neutral']:.1%} | "
                  f"Pos={result['probabilities']['positive']:.1%}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "../../models/custom_konkani_model"
    test_model(model_dir)
