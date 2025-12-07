#!/usr/bin/env python3
"""
Validate UTF-8 Encoding and Unicode Normalization

This script validates that all dataset files are properly encoded in UTF-8
and that Unicode normalization is consistent across the dataset.
"""

import os
import json
import pandas as pd
import unicodedata
from collections import Counter
from pathlib import Path


def check_file_encoding(filepath: str) -> dict:
    """Check if file is valid UTF-8"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'valid_utf8': True,
            'error': None,
            'size_bytes': len(content.encode('utf-8'))
        }
    except UnicodeDecodeError as e:
        return {
            'valid_utf8': False,
            'error': str(e),
            'size_bytes': None
        }


def check_unicode_normalization(text: str) -> dict:
    """Check Unicode normalization form"""
    nfc = unicodedata.normalize('NFC', text)
    nfd = unicodedata.normalize('NFD', text)
    nfkc = unicodedata.normalize('NFKC', text)
    nfkd = unicodedata.normalize('NFKD', text)
    
    return {
        'is_nfc': text == nfc,
        'is_nfd': text == nfd,
        'is_nfkc': text == nfkc,
        'is_nfkd': text == nfkd,
        'original_length': len(text),
        'nfc_length': len(nfc),
        'nfd_length': len(nfd)
    }


def detect_mojibake(text: str) -> dict:
    """Detect common mojibake patterns"""
    mojibake_indicators = [
        '�',  # Replacement character
        '\ufffd',  # Unicode replacement character
        'Ã', 'â', 'Â',  # Common UTF-8 misinterpretation
    ]
    
    found = []
    for indicator in mojibake_indicators:
        if indicator in text:
            found.append(indicator)
    
    return {
        'has_mojibake': len(found) > 0,
        'indicators_found': found
    }


def validate_devanagari_characters(text: str) -> dict:
    """Validate Devanagari Unicode characters"""
    devanagari_range = range(0x0900, 0x097F + 1)  # Devanagari Unicode block
    
    devanagari_chars = []
    invalid_chars = []
    
    for char in text:
        code_point = ord(char)
        if code_point in devanagari_range:
            devanagari_chars.append(char)
        elif not char.isascii() and not char.isspace():
            # Non-ASCII, non-Devanagari, non-whitespace
            invalid_chars.append((char, hex(code_point)))
    
    return {
        'has_devanagari': len(devanagari_chars) > 0,
        'devanagari_count': len(devanagari_chars),
        'invalid_chars': invalid_chars[:10]  # First 10 invalid
    }


def validate_dataset(filepath: str) -> dict:
    """Comprehensive validation of dataset file"""
    print(f"\nValidating: {filepath}")
    print("-" * 60)
    
    # Check file encoding
    encoding_check = check_file_encoding(filepath)
    print(f"  UTF-8 Valid: {encoding_check['valid_utf8']}")
    
    if not encoding_check['valid_utf8']:
        print(f"  ERROR: {encoding_check['error']}")
        return {
            'file': filepath,
            'valid': False,
            'error': encoding_check['error']
        }
    
    # Load dataset
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='utf-8')
        elif filepath.endswith('.jsonl'):
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        else:
            print(f"  Skipping: Unsupported format")
            return None
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return {
            'file': filepath,
            'valid': False,
            'error': str(e)
        }
    
    print(f"  Rows: {len(df):,}")
    
    # Check text columns
    text_columns = [col for col in df.columns if 'text' in col.lower() or 'devanagari' in col.lower()]
    print(f"  Text columns: {text_columns}")
    
    results = {
        'file': filepath,
        'valid': True,
        'rows': len(df),
        'encoding': encoding_check,
        'columns': {}
    }
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        print(f"\n  Analyzing column: {col}")
        
        # Sample texts for analysis
        sample_size = min(1000, len(df))
        sample_texts = df[col].dropna().sample(n=sample_size, random_state=42).tolist()
        
        # Check normalization
        norm_forms = Counter()
        mojibake_count = 0
        devanagari_texts = 0
        invalid_char_examples = []
        
        for text in sample_texts:
            if not isinstance(text, str):
                continue
            
            # Normalization
            norm = check_unicode_normalization(text)
            if norm['is_nfc']:
                norm_forms['NFC'] += 1
            elif norm['is_nfd']:
                norm_forms['NFD'] += 1
            else:
                norm_forms['Mixed'] += 1
            
            # Mojibake
            mojibake = detect_mojibake(text)
            if mojibake['has_mojibake']:
                mojibake_count += 1
            
            # Devanagari validation
            if 'devanagari' in col.lower():
                dev_check = validate_devanagari_characters(text)
                if dev_check['has_devanagari']:
                    devanagari_texts += 1
                if dev_check['invalid_chars']:
                    invalid_char_examples.extend(dev_check['invalid_chars'])
        
        print(f"    Normalization: {dict(norm_forms)}")
        print(f"    Mojibake detected: {mojibake_count}/{sample_size}")
        
        if 'devanagari' in col.lower():
            print(f"    Valid Devanagari: {devanagari_texts}/{sample_size}")
            if invalid_char_examples:
                print(f"    Invalid chars found: {invalid_char_examples[:5]}")
        
        results['columns'][col] = {
            'normalization': dict(norm_forms),
            'mojibake_count': mojibake_count,
            'sample_size': sample_size,
            'devanagari_valid': devanagari_texts if 'devanagari' in col.lower() else None
        }
    
    return results


def main():
    """Main validation routine"""
    print("="*60)
    print("UTF-8 ENCODING & UNICODE VALIDATION")
    print("="*60)
    
    # Files to validate
    files_to_check = [
        'custom_konkani_sentiment.csv',
        'custom_konkani_sentiment_fixed.csv',
        'custom_konkani_sentiment.jsonl',
        'konkani_sentiment.csv'
    ]
    
    all_results = []
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"\nSkipping {filepath} (not found)")
            continue
        
        result = validate_dataset(filepath)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_valid = all(r['valid'] for r in all_results)
    
    if all_valid:
        print("\n✓ All files are valid UTF-8")
        print("✓ No encoding errors detected")
        
        # Check normalization consistency
        all_nfc = all(
            all(col_data['normalization'].get('NFC', 0) == col_data['sample_size'] 
                for col_data in r['columns'].values())
            for r in all_results
        )
        
        if all_nfc:
            print("✓ All text is in NFC normalization (consistent)")
        else:
            print("⚠️  Mixed normalization forms detected")
        
        # Check mojibake
        total_mojibake = sum(
            sum(col_data['mojibake_count'] for col_data in r['columns'].values())
            for r in all_results
        )
        
        if total_mojibake == 0:
            print("✓ No mojibake detected")
        else:
            print(f"⚠️  Mojibake detected in {total_mojibake} samples")
    else:
        print("\n✗ Some files have encoding errors")
        for r in all_results:
            if not r['valid']:
                print(f"  {r['file']}: {r.get('error', 'Unknown error')}")
    
    # Save report
    report = {
        'validation_date': pd.Timestamp.now().isoformat(),
        'files_checked': len(all_results),
        'all_valid': all_valid,
        'results': all_results
    }
    
    report_file = 'encoding_validation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Validation report saved: {report_file}")
    
    return all_valid


if __name__ == "__main__":
    is_valid = main()
    exit(0 if is_valid else 1)
