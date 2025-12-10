"""Translation Model Wrapper using NLLB"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path


class TranslationModel:
    """Wrapper for NLLB translation model"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize translation model
        
        Args:
            model_path: Path to finetuned model (None for base NLLB)
            device: Device to use
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        # Use finetuned model if available, otherwise base NLLB
        if model_path and Path(model_path).exists():
            print(f"Loading finetuned NLLB from {model_path}")
            model_name = model_path
        else:
            print("Loading base NLLB model")
            model_name = "facebook/nllb-200-distilled-600M"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Language codes
        self.konkani_code = "kok_Deva"
        self.english_code = "eng_Latn"
    
    def translate(self, text, src_lang="kok_Deva", tgt_lang="eng_Latn"):
        """
        Translate text
        
        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
        
        Returns:
            translation: Translated text
        """
        self.tokenizer.src_lang = src_lang
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=200,
                num_beams=5,
                early_stopping=True
            )
        
        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        return translation
    
    def konkani_to_english(self, text):
        """Translate Konkani to English"""
        return self.translate(text, self.konkani_code, self.english_code)
    
    def english_to_konkani(self, text):
        """Translate English to Konkani"""
        return self.translate(text, self.english_code, self.konkani_code)
