"""Emotion Detection Model Wrapper"""
import torch
from pathlib import Path

from .konkani_custom_emotion import CustomEmotionModel


class EmotionModel:
    """Wrapper for emotion detection model"""
    
    EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    
    def __init__(self, checkpoint_path=None, device=None):
        """Initialize emotion model"""
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        # Resolve checkpoint path
        if checkpoint_path is None:
            base_dir = Path(__file__).parent.parent.parent
            possible_paths = [
                base_dir / "checkpoints" / "emotion_model" / "emotion_model_mac.pt",
                base_dir / "checkpoints" / "emotion_model" / "emotion_model_best.pt",
                Path("checkpoints/emotion_model/emotion_model_mac.pt"),
                Path("checkpoints/emotion_model/emotion_model_best.pt"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "Could not find emotion model checkpoint. Tried:\n" + 
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get vocab_size from checkpoint
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
        elif 'config' in checkpoint and 'vocab_size' in checkpoint['config']:
            vocab_size = checkpoint['config']['vocab_size']
        elif 'vocab' in checkpoint:
            vocab_size = len(checkpoint['vocab'])
        else:
            raise ValueError("Could not find vocab_size in checkpoint")
        
        # Create model
        self.model = CustomEmotionModel(
            vocab_size=vocab_size,
            num_emotions=len(self.EMOTIONS)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load vocabulary
        if 'char_to_idx' in checkpoint:
            self.char_to_idx = checkpoint['char_to_idx']
        elif 'vocab' in checkpoint:
            # Build char_to_idx from vocab dict
            self.char_to_idx = checkpoint['vocab']
        else:
            raise ValueError("Could not find vocabulary in checkpoint")
    
    def predict(self, text):
        """
        Predict emotion from text
        
        Args:
            text: Konkani text
        
        Returns:
            emotion: Predicted emotion label
            confidence: Confidence score
            all_scores: Dictionary of all emotion scores
        """
        # Tokenize text
        tokens = [self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 1)) for c in text]
        
        # Create tensor
        input_ids = torch.tensor([tokens]).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Predict
        with torch.no_grad():
            predictions, probabilities, _ = self.model.predict(input_ids, attention_mask)
        
        # Get results
        emotion_idx = predictions[0].item()
        emotion = self.EMOTIONS[emotion_idx]
        confidence = probabilities[0][emotion_idx].item()
        
        # All scores
        all_scores = {
            self.EMOTIONS[i]: probabilities[0][i].item()
            for i in range(len(self.EMOTIONS))
        }
        
        return emotion, confidence, all_scores
