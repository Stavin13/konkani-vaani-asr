from datasets import load_dataset
import soundfile as sf

def inspect_konkani_dataset():
    print("Downloading Konkani (Devanagari) dataset from Google FLEURS...")
    # Load the dataset (streaming mode to avoid downloading everything immediately)
    dataset = load_dataset("google/fleurs", "kon_deva", split="train", streaming=True)
    
    print("\nDataset loaded successfully! Here are the first 3 examples:")
    print("-" * 50)
    
    # Iterate through the first 3 examples
    for i, example in enumerate(dataset.take(3)):
        print(f"Example {i+1}:")
        print(f"  ID: {example['id']}")
        print(f"  Transcript: {example['transcription']}")
        print(f"  Audio Info: {example['audio']}")
        print("-" * 50)

if __name__ == "__main__":
    inspect_konkani_dataset()
