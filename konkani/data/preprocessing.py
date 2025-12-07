"""
Data preprocessing utilities for Konkani text.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Clean and normalize Konkani text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep Devanagari and basic punctuation
    # This is a basic implementation - can be enhanced
    
    return text.strip()


def normalize_devanagari(text: str) -> str:
    """
    Normalize Devanagari text.
    
    Args:
        text: Input text in Devanagari
        
    Returns:
        Normalized text
    """
    # Add normalization rules for Devanagari if needed
    # For now, just return cleaned text
    return clean_text(text)


def tokenize_simple(text: str) -> List[str]:
    """
    Simple tokenization for Konkani text.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Keep Devanagari and Roman characters
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens
