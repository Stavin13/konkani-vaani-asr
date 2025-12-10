#!/usr/bin/env python3
"""
Download checkpoint from Kaggle notebook output
"""

import os
import subprocess
import sys
from pathlib import Path

def download_kaggle_checkpoint():
    """Download the trained checkpoint from Kaggle"""
    
    # Your Kaggle notebook reference
    notebook_ref = "stavinfernandes/kaggle-train-10k-scripts1-fixed"
    
    # Create download directory
    download_dir = Path("kaggle_new_checkpoint")
    download_dir.mkdir(exist_ok=True)
    
    print(f"📥 Downloading checkpoint from Kaggle notebook: {notebook_ref}")
    print(f"📁 Download directory: {download_dir}")
    
    try:
        # Download using kaggle CLI
        cmd = [
            "kaggle", "kernels", "output", 
            notebook_ref, 
            "-p", str(download_dir)
        ]
        
        print(f"🔄 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Download successful!")
            
            # List downloaded files
            print("\n📋 Downloaded files:")
            for file in download_dir.rglob("*"):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  📄 {file.name} ({size_mb:.1f} MB)")
            
            # Look for checkpoint files
            checkpoint_files = list(download_dir.glob("**/*.pt"))
            if checkpoint_files:
                print(f"\n🎯 Found {len(checkpoint_files)} checkpoint files:")
                for ckpt in checkpoint_files:
                    print(f"  🔥 {ckpt}")
                
                # Copy best checkpoint to main checkpoints directory
                main_ckpt_dir = Path("checkpoints")
                main_ckpt_dir.mkdir(exist_ok=True)
                
                for ckpt in checkpoint_files:
                    if "best" in ckpt.name.lower():
                        dest = main_ckpt_dir / f"kaggle_best_model_scripts1.pt"
                        import shutil
                        shutil.copy2(ckpt, dest)
                        print(f"✅ Copied best model to: {dest}")
                        break
                else:
                    # Copy the first checkpoint if no "best" found
                    if checkpoint_files:
                        dest = main_ckpt_dir / f"kaggle_checkpoint_scripts1.pt"
                        import shutil
                        shutil.copy2(checkpoint_files[0], dest)
                        print(f"✅ Copied checkpoint to: {dest}")
            else:
                print("⚠️  No .pt checkpoint files found")
                
        else:
            print(f"❌ Download failed!")
            print(f"Error: {result.stderr}")
            print(f"Output: {result.stdout}")
            
            # Check if kaggle CLI is installed
            try:
                subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
                print("✅ Kaggle CLI is installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Kaggle CLI not found. Install with: pip install kaggle")
                print("📖 Setup guide: https://github.com/Kaggle/kaggle-api")
                return False
                
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_kaggle_checkpoint()
    if success:
        print("\n🎉 Checkpoint download completed!")
        print("📁 Check the 'kaggle_new_checkpoint' directory for all files")
        print("🔥 Best model copied to 'checkpoints/' directory")
    else:
        print("\n💡 Alternative: Download manually from Kaggle and place in 'checkpoints/' directory")