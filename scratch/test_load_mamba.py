import sys
from pathlib import Path
import torch

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "mamba"))

try:
    from train_custom_mamba import TinyMambaCorrectorModel, KonkaniCharTokenizer
    print("Import successful!")
    
    # Load model config and checkpoint
    checkpoint_path = BASE / "mamba" / "best_model_mamba_test.pt"
    vocab_path = BASE / "data/vocab.json"
    
    tokenizer = KonkaniCharTokenizer(str(vocab_path))
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.char2idx)}")
    
    device = "cpu"
    state_dict = torch.load(checkpoint_path, map_location=device)
    config = state_dict.get('config', {})
    print(f"Config from checkpoint: {config}")
    
    # Load model with exact config from checkpoint if available
    d_model = config.get("d_model", 256)
    n_layers = config.get("n_layers", 6)
    d_state = config.get("d_state", 16)
    d_conv = config.get("d_conv", 4)
    expand = config.get("expand", 2)
    dropout = config.get("dropout", 0.1)
    
    model = TinyMambaCorrectorModel(
        vocab_size=83, # VOCAB_SIZE in train_custom_mamba.py is 83
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout
    )
    
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict['model_state'].items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Test generation
    prompt = "ASR output text"
    src_ids = tokenizer.encode(prompt) + [tokenizer.sep_id]
    src_tensor = torch.tensor([src_ids], device=device)
    # Use generate from model
    out_ids = model.generate(src_tensor, max_new=20)
    decoded = tokenizer.decode(out_ids)
    print(f"Generated text: {decoded}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
