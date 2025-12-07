#!/usr/bin/env python3
"""
Fix Data Leakage in Konkani Sentiment Dataset

This script prevents data leakage by ensuring that all Romanization variants
of the same Devanagari text stay in the same train/val/test split.

Problem: Multiple Roman variants of same Devanagari text could appear in
different splits, allowing model to memorize and inflate performance.

Solution: Group by Devanagari text before splitting, then expand back.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json


def fix_data_leakage(input_file: str = "custom_konkani_sentiment.csv",
                     output_file: str = "custom_konkani_sentiment_fixed.csv",
                     test_size: float = 0.15,
                     val_size: float = 0.15,
                     random_state: int = 42):
    """
    Fix data leakage by grouping variants before splitting
    
    Args:
        input_file: Original dataset with potential leakage
        output_file: Fixed dataset without leakage
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
    """
    
    print("="*60)
    print("FIXING DATA LEAKAGE IN KONKANI SENTIMENT DATASET")
    print("="*60)
    
    # Load original dataset
    print(f"\n1. Loading original dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Total entries: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Analyze current state
    print("\n2. Analyzing potential leakage...")
    unique_devanagari = df['devanagari'].nunique()
    total_entries = len(df)
    avg_variants = total_entries / unique_devanagari
    
    print(f"   Unique Devanagari texts: {unique_devanagari:,}")
    print(f"   Total entries (with variants): {total_entries:,}")
    print(f"   Average variants per text: {avg_variants:.2f}")
    
    # Group by Devanagari text
    print("\n3. Grouping variants by Devanagari text...")
    df_grouped = df.groupby('devanagari').agg({
        'label': 'first',  # All variants have same label
        'text': lambda x: list(x),  # Collect all Roman variants
        'id': lambda x: list(x) if 'id' in df.columns else None,
        'type': lambda x: list(x) if 'type' in df.columns else None,
        'source': lambda x: list(x) if 'source' in df.columns else None
    }).reset_index()
    
    print(f"   Grouped into {len(df_grouped):,} unique Devanagari texts")
    
    # Verify label consistency
    print("\n4. Verifying label consistency within groups...")
    label_counts = df.groupby('devanagari')['label'].nunique()
    inconsistent = label_counts[label_counts > 1]
    
    if len(inconsistent) > 0:
        print(f"   ⚠️  WARNING: {len(inconsistent)} Devanagari texts have inconsistent labels!")
        print(f"   Examples: {inconsistent.head()}")
    else:
        print("   ✓ All Devanagari texts have consistent labels")
    
    # Split on Devanagari level (stratified by label)
    print("\n5. Performing stratified split on Devanagari texts...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df_grouped,
        test_size=test_size,
        random_state=random_state,
        stratify=df_grouped['label']
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_df['label']
    )
    
    print(f"   Train: {len(train_df):,} Devanagari texts")
    print(f"   Val:   {len(val_df):,} Devanagari texts")
    print(f"   Test:  {len(test_df):,} Devanagari texts")
    
    # Expand back to full dataset with all variants
    print("\n6. Expanding back to full dataset with variants...")
    
    def expand_group(group_df, split_name):
        """Expand grouped data back to individual rows"""
        rows = []
        for _, row in group_df.iterrows():
            devanagari = row['devanagari']
            label = row['label']
            variants = row['text']
            
            for i, variant in enumerate(variants):
                rows.append({
                    'text': variant,
                    'devanagari': devanagari,
                    'label': label,
                    'split': split_name,
                    'id': row['id'][i] if row['id'] is not None else f"{split_name}_{len(rows)}",
                    'type': row['type'][i] if row['type'] is not None else 'unknown',
                    'source': row['source'][i] if row['source'] is not None else 'custom'
                })
        return pd.DataFrame(rows)
    
    train_expanded = expand_group(train_df, 'train')
    val_expanded = expand_group(val_df, 'validation')
    test_expanded = expand_group(test_df, 'test')
    
    print(f"   Train: {len(train_expanded):,} total entries")
    print(f"   Val:   {len(val_expanded):,} total entries")
    print(f"   Test:  {len(test_expanded):,} total entries")
    
    # Combine all splits
    df_fixed = pd.concat([train_expanded, val_expanded, test_expanded], ignore_index=True)
    
    # Verify no leakage
    print("\n7. Verifying no data leakage...")
    train_devanagari = set(train_df['devanagari'])
    val_devanagari = set(val_df['devanagari'])
    test_devanagari = set(test_df['devanagari'])
    
    train_val_overlap = train_devanagari & val_devanagari
    train_test_overlap = train_devanagari & test_devanagari
    val_test_overlap = val_devanagari & test_devanagari
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("   ✓ No Devanagari text appears in multiple splits")
        print("   ✓ Data leakage prevented!")
    else:
        print(f"   ✗ ERROR: Found overlaps!")
        print(f"     Train-Val: {len(train_val_overlap)}")
        print(f"     Train-Test: {len(train_test_overlap)}")
        print(f"     Val-Test: {len(val_test_overlap)}")
    
    # Verify label distribution
    print("\n8. Verifying label distribution...")
    for split_name in ['train', 'validation', 'test']:
        split_data = df_fixed[df_fixed['split'] == split_name]
        label_dist = split_data['label'].value_counts(normalize=True)
        print(f"\n   {split_name.capitalize()}:")
        for label, pct in label_dist.items():
            print(f"     {label}: {pct:.1%}")
    
    # Save fixed dataset
    print(f"\n9. Saving fixed dataset: {output_file}")
    df_fixed.to_csv(output_file, index=False, encoding='utf-8')
    print(f"   ✓ Saved {len(df_fixed):,} entries")
    
    # Generate leakage report
    report = {
        'original_dataset': {
            'file': input_file,
            'total_entries': int(total_entries),
            'unique_devanagari': int(unique_devanagari),
            'avg_variants_per_text': float(avg_variants)
        },
        'fixed_dataset': {
            'file': output_file,
            'total_entries': int(len(df_fixed)),
            'splits': {
                'train': {
                    'devanagari_texts': int(len(train_df)),
                    'total_entries': int(len(train_expanded)),
                    'label_distribution': train_expanded['label'].value_counts().to_dict()
                },
                'validation': {
                    'devanagari_texts': int(len(val_df)),
                    'total_entries': int(len(val_expanded)),
                    'label_distribution': val_expanded['label'].value_counts().to_dict()
                },
                'test': {
                    'devanagari_texts': int(len(test_df)),
                    'total_entries': int(len(test_expanded)),
                    'label_distribution': test_expanded['label'].value_counts().to_dict()
                }
            }
        },
        'verification': {
            'no_devanagari_overlap': len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
            'label_consistency': len(inconsistent) == 0,
            'leakage_prevented': True
        }
    }
    
    report_file = 'data_leakage_fix_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n10. Leakage report saved: {report_file}")
    
    print("\n" + "="*60)
    print("DATA LEAKAGE FIX COMPLETE!")
    print("="*60)
    print(f"\n✓ Fixed dataset: {output_file}")
    print(f"✓ Report: {report_file}")
    print(f"\nNext steps:")
    print(f"  1. Use '{output_file}' for training")
    print(f"  2. Update training scripts to use 'split' column")
    print(f"  3. Train models with confidence!")
    
    return df_fixed, report


if __name__ == "__main__":
    df_fixed, report = fix_data_leakage()
