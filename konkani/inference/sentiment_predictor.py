"""
Sentiment prediction for Konkani.
"""

import torch
from pathlib import Path
from typing import Dict, Union

from ..core.tokenizer import KonkaniTokenizer
from ..models.sentiment.bilstm import CustomKonkaniSentimentModel
from ..utils.io import load_json


class SentimentPredictor:
    """Predictor for Konkani sentiment analysis"""
    
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        
        # Load tokenizer
        self.tokenizer = KonkaniTokenizer.load(self.model_dir / "tokenizer.pkl")
        
        # Load label map
        label_map_data = load_json(self.model_dir / "label_map.json")
        self.id2label = {int(k): v for k, v in label_map_data['id2label'].items()}
        
        # Load model
        checkpoint = torch.load(
            self.model_dir / "best_model.pt",
            map_location='cpu'
        )
        
        config = checkpoint['model_config']
        self.model = CustomKonkaniSentimentModel(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with label, confidence, and probabilities
        """
        # Encode text
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attention_weights = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return {
            'label': self.id2label[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                self.id2label[i]: float(prob)
                for i, prob in enumerate(probabilities.cpu().numpy())
            }
        }
