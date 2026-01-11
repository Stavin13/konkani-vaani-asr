import json
from collections import Counter

def script_stats(manifest_path):
    with open(manifest_path) as f:
        lines = [json.loads(l) for l in f]
    
    script_dist = Counter()
    char_samples = []
    
    for item in lines[:500]:  # Sample 500
        text = item['text']
        for char in text:
            cp = ord(char)
            if 0x0900 <= cp <= 0x097F:
                script_dist['Devanagari'] += 1
            elif 0x0600 <= cp <= 0x06FF:
                script_dist['Arabic'] += 1
            elif 0x0041 <= cp <= 0x007A:
                script_dist['Latin'] += 1
            elif char not in ' .?!,-':
                script_dist['Other'] += 1
    
    total = sum(script_dist.values())
    print(f"\n📊 SCRIPT DISTRIBUTION in {manifest_path}:")
    for script, count in script_dist.most_common():
        print(f"  {script}: {count} chars ({count/total*100:.1f}%)")
    
    return script_dist

# Run on ALL your manifests
print("="*50)
train_stats = script_stats('train_manifest.json')
print("="*50)
val_stats = script_stats('val_manifest.json')