"""
Fix audio file paths in manifest files for Colab
Converts Mac paths to Colab paths
"""

import json
import sys
import os

def fix_manifest(input_file, output_file=None):
    """Fix paths in a manifest file"""
    if output_file is None:
        output_file = input_file
    
    fixed_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Get the old path
            old_path = data['audio_filepath']
            
            # Extract just the filename part after 'konkani/'
            if 'konkani/' in old_path:
                # Get everything after 'konkani/'
                relative_path = old_path.split('konkani/', 1)[1]
            else:
                # Fallback: get the relative path from data/
                relative_path = old_path.split('data/', 1)[1] if 'data/' in old_path else old_path
            
            # Create new path (assuming we're in the project root)
            new_path = relative_path
            
            # Update the path
            data['audio_filepath'] = new_path
            fixed_data.append(data)
    
    # Write fixed data
    with open(output_file, 'w') as f:
        for data in fixed_data:
            f.write(json.dumps(data) + '\n')
    
    print(f"✅ Fixed {len(fixed_data)} entries in {input_file}")
    if fixed_data:
        print(f"   Example: {fixed_data[0]['audio_filepath']}")

if __name__ == "__main__":
    # Fix all manifest files
    manifests = [
        'data/konkani-asr-v0/splits/manifests/train.json',
        'data/konkani-asr-v0/splits/manifests/val.json',
        'data/konkani-asr-v0/splits/manifests/test.json'
    ]
    
    for manifest in manifests:
        if os.path.exists(manifest):
            fix_manifest(manifest)
        else:
            print(f"⚠️  Skipping {manifest} (not found)")
    
    print("\n✅ All manifests fixed!")
