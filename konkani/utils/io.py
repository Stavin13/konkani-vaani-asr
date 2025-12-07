"""
I/O utilities for Konkani NLP.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict[str, Any], path: Path, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save to
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_checksum(filepath: Path) -> str:
    """
    Generate SHA256 checksum for file.
    
    Args:
        filepath: Path to file
        
    Returns:
        SHA256 checksum as hex string
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_checksums(directory: Path, patterns: list = None) -> Dict[str, str]:
    """
    Generate checksums for all files in a directory.
    
    Args:
        directory: Directory to scan
        patterns: List of file patterns to include (e.g., ['*.pt', '*.pkl'])
        
    Returns:
        Dictionary mapping filenames to checksums
    """
    directory = Path(directory)
    checksums = {}
    
    if patterns is None:
        patterns = ['*']
    
    for pattern in patterns:
        for filepath in directory.glob(pattern):
            if filepath.is_file():
                checksums[filepath.name] = generate_checksum(filepath)
    
    return checksums
