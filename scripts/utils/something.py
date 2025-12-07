import json
import pandas as pd
from collections import Counter

def validate_custom_dataset():
    """Validate the custom Konkani dataset"""
    
    print("VALIDATING CUSTOM KONKANI DATASET")
    print("="*50)
    
    # Load the dataset
    with open('custom_konkani_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 1. Check metadata
    print("\n1. METADATA:")
    print(f"   Dataset: {dataset['metadata']['name']}")
    print(f"   Version: {dataset['metadata']['version']}")
    print(f"   Created: {dataset['metadata']['created']}")
    print(f"   Unique ID: {dataset['metadata']['unique_id']}")
    
    # 2. Check word counts
    print("\n2. WORD COUNTS:")
    for sentiment, count in dataset['word_count'].items():
        print(f"   {sentiment}: {count} unique words")
    
    # 3. Check quality metrics
    print("\n3. QUALITY METRICS:")
    metrics = dataset['quality_metrics']
    print(f"   Unique Devanagari words: {metrics['unique_devanagari_words']}")
    print(f"   Unique sentences: {metrics['unique_sentences']}")
    print(f"   Avg variants per word: {metrics['avg_variants_per_word']:.2f}")
    print(f"   Avg variants per sentence: {metrics['avg_variants_per_sentence']:.2f}")
    
    # 4. Load CSV and analyze
    df = pd.read_csv('custom_konkani_sentiment.csv')
    
    print("\n4. CSV ANALYSIS:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # 5. Check label distribution
    print("\n5. LABEL DISTRIBUTION:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count:,} ({percentage:.1f}%)")
    
    # 6. Check for duplicates
    print("\n6. DUPLICATE CHECK")
    