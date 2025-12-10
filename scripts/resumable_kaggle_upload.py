#!/usr/bin/env python3
"""
Resumable Kaggle dataset upload with retry logic
"""
import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi

def upload_with_retry(dataset_path, dataset_slug, max_retries=3, retry_delay=60):
    """
    Upload dataset with automatic retry on failure
    
    Args:
        dataset_path: Path to dataset folder or zip
        dataset_slug: Your Kaggle dataset slug (username/dataset-name)
        max_retries: Number of retry attempts
        retry_delay: Seconds to wait between retries
    """
    api = KaggleApi()
    api.authenticate()
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n{'='*60}")
            print(f"Upload attempt {attempt}/{max_retries}")
            print(f"{'='*60}")
            
            # Create or update dataset
            if os.path.isdir(dataset_path):
                print(f"Uploading directory: {dataset_path}")
                api.dataset_create_version(
                    dataset_path,
                    version_notes=f"Upload attempt {attempt}",
                    dir_mode='zip',
                    quiet=False
                )
            else:
                print(f"Uploading file: {dataset_path}")
                api.dataset_create_version(
                    dataset_path,
                    version_notes=f"Upload attempt {attempt}",
                    quiet=False
                )
            
            print("\n✓ Upload successful!")
            return True
            
        except Exception as e:
            print(f"\n✗ Upload failed: {str(e)}")
            
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"\n✗ All {max_retries} attempts failed")
                return False
    
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python resumable_kaggle_upload.py <dataset_path> <username/dataset-slug>")
        print("\nExample:")
        print("  python resumable_kaggle_upload.py konkani_10k_dataset.zip yourusername/konkani-10k")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    dataset_slug = sys.argv[2]
    
    if not os.path.exists(dataset_path):
        print(f"Error: Path not found: {dataset_path}")
        sys.exit(1)
    
    success = upload_with_retry(dataset_path, dataset_slug)
    sys.exit(0 if success else 1)
