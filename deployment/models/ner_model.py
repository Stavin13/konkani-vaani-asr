"""NER Model Wrapper"""
import torch
from pathlib import Path

from .konkani_ner import KonkaniNER


class NERModel:
    """Wrapper for NER model"""
    
    TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    
    def __init__(self, checkpoint_path=None, device=None):
        """Initialize NER model"""
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
                base_dir / "checkpoints" / "ner" / "best_ner_model.pt",
                base_dir / "checkpoints" / "ner" / "ner_model.pt",
                Path("checkpoints/ner/best_ner_model.pt"),
                Path("checkpoints/ner/ner_model.pt"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "Could not find NER model checkpoint. Tried:\n" + 
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load vocabularies from checkpoint or from file
        if 'word_to_idx' in checkpoint and 'char_to_idx' in checkpoint and 'idx_to_tag' in checkpoint:
            self.word_to_idx = checkpoint['word_to_idx']
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_tag = checkpoint['idx_to_tag']
        else:
            # Load from vocabularies.json file
            import json
            base_dir = Path(__file__).parent.parent.parent
            vocab_paths = [
                base_dir / "checkpoints" / "ner" / "vocabularies.json",
                Path("checkpoints/ner/vocabularies.json"),
            ]
            
            vocab_file = None
            for vp in vocab_paths:
                if vp.exists():
                    vocab_file = vp
                    break
            
            if vocab_file is None:
                raise FileNotFoundError("Could not find vocabularies.json file for NER model")
            
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Map from different naming conventions
            self.word_to_idx = vocab_data.get('word_to_idx') or vocab_data.get('word2id', {})
            self.char_to_idx = vocab_data.get('char_to_idx') or vocab_data.get('char2id', {})
            
            # Create idx_to_tag from TAGS
            self.idx_to_tag = {i: tag for i, tag in enumerate(self.TAGS)}
        
        # Create model
        self.model = KonkaniNER(
            vocab_size=len(self.word_to_idx),
            char_vocab_size=len(self.char_to_idx),
            num_tags=len(self.idx_to_tag)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        """
        Extract named entities from text
        
        Args:
            text: Konkani text
        
        Returns:
            entities: List of (entity_text, entity_type, start, end)
        """
        # Tokenize (simple whitespace tokenization)
        words = text.split()
        
        # Convert to indices
        word_ids = [self.word_to_idx.get(w, self.word_to_idx.get('<UNK>', 1)) for w in words]
        
        # Character IDs (simplified - just use first 20 chars)
        char_ids = []
        for word in words:
            chars = [self.char_to_idx.get(c, 1) for c in word[:20]]
            chars += [0] * (20 - len(chars))  # Pad to 20
            char_ids.append(chars)
        
        # Create tensors
        word_ids_tensor = torch.tensor([word_ids]).to(self.device)
        char_ids_tensor = torch.tensor([char_ids]).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(word_ids_tensor, char_ids_tensor)
        
        # Decode predictions
        pred_tags = predictions[0]  # First (and only) sequence
        
        # Extract entities
        entities = []
        current_entity = None
        current_type = None
        start_idx = None
        
        for i, (word, tag_idx) in enumerate(zip(words, pred_tags)):
            tag = self.idx_to_tag[tag_idx]
            
            if tag.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    entities.append((
                        ' '.join(current_entity),
                        current_type,
                        start_idx,
                        i - 1
                    ))
                
                # Start new entity
                current_entity = [word]
                current_type = tag[2:]  # Remove 'B-'
                start_idx = i
            
            elif tag.startswith('I-') and current_entity:
                # Continue entity
                current_entity.append(word)
            
            else:
                # End entity
                if current_entity:
                    entities.append((
                        ' '.join(current_entity),
                        current_type,
                        start_idx,
                        i - 1
                    ))
                    current_entity = None
                    current_type = None
        
        # Don't forget last entity
        if current_entity:
            entities.append((
                ' '.join(current_entity),
                current_type,
                start_idx,
                len(words) - 1
            ))
        
        return entities
