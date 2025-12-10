"""
Extract Training Data from Raw Kaggle Logs
Handles the complete raw output including timestamps
"""

import re
import json
from pathlib import Path
from collections import defaultdict


def extract_all_training_data(log_file):
    """Extract all training data from raw Kaggle logs"""
    
    print(f"Reading log file: {log_file}")
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines):,}")
    
    # Patterns
    epoch_progress_pattern = r'Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s+\[[^\]]+loss=([\d.]+),\s*ctc=([\d.]+)\]'
    validation_pattern = r'Validation:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)'
    
    # Patterns for epoch summaries (with timestamps)
    # Format: "655.4s 294 Train Loss: 14.3275 (CTC: 17.1323)"
    train_loss_pattern = r'[\d.]+s\s+\d+\s+Train Loss:\s*([\d.]+)\s*\(CTC:\s*([\d.]+)\)'
    val_loss_pattern = r'[\d.]+s\s+\d+\s+Val Loss:\s*([\d.]+)\s*\(CTC:\s*([\d.]+)\)'
    epoch_header_pattern = r'[\d.]+s\s+\d+\s+Epoch\s+(\d+)/(\d+)'
    
    # Storage
    epoch_data = defaultdict(lambda: {'train_losses': [], 'train_ctc': [], 'steps': []})
    epoch_summaries = []
    
    # Parse line by line
    current_epoch = None
    pending_train_loss = None
    pending_val_loss = None
    
    for i, line in enumerate(lines):
        # Check for epoch header
        epoch_match = re.search(epoch_header_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        
        # Check for train loss summary
        train_match = re.search(train_loss_pattern, line)
        if train_match:
            pending_train_loss = {
                'train_loss': float(train_match.group(1)),
                'train_ctc': float(train_match.group(2))
            }
            continue
        
        # Check for val loss summary
        val_match = re.search(val_loss_pattern, line)
        if val_match and pending_train_loss and current_epoch:
            val_loss = float(val_match.group(1))
            val_ctc = float(val_match.group(2))
            
            # Complete epoch summary
            epoch_summaries.append({
                'epoch': current_epoch,
                'train_loss': pending_train_loss['train_loss'],
                'train_ctc': pending_train_loss['train_ctc'],
                'val_loss': val_loss,
                'val_ctc': val_ctc
            })
            
            pending_train_loss = None
            continue
        
        # Check for epoch progress
        progress_match = re.search(epoch_progress_pattern, line)
        if progress_match:
            epoch = int(progress_match.group(1))
            step = int(progress_match.group(2))
            total_steps = int(progress_match.group(3))
            loss = float(progress_match.group(4))
            ctc = float(progress_match.group(5))
            
            epoch_data[epoch]['train_losses'].append(loss)
            epoch_data[epoch]['train_ctc'].append(ctc)
            epoch_data[epoch]['steps'].append(step)
            epoch_data[epoch]['total_steps'] = total_steps
    
    print(f"\n✅ Extraction complete!")
    print(f"   Epochs with progress data: {len(epoch_data)}")
    print(f"   Epoch summaries found: {len(epoch_summaries)}")
    
    if epoch_summaries:
        print(f"   Epoch range: {min(s['epoch'] for s in epoch_summaries)} to {max(s['epoch'] for s in epoch_summaries)}")
    
    return {
        'epoch_data': dict(epoch_data),
        'summaries': epoch_summaries
    }


def save_extracted_data(data, output_file='extracted_training_data.json'):
    """Save extracted data to JSON for later use"""
    output_path = Path(output_file)
    
    # Convert to serializable format
    serializable_data = {
        'epoch_data': {
            str(k): {
                'train_losses': v['train_losses'],
                'train_ctc': v['train_ctc'],
                'steps': v['steps'],
                'total_steps': v.get('total_steps', 0)
            }
            for k, v in data['epoch_data'].items()
        },
        'summaries': data['summaries']
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"\n✅ Saved extracted data to: {output_path}")
    return output_path


def print_summary(data):
    """Print summary of extracted data"""
    epoch_data = data['epoch_data']
    summaries = data['summaries']
    
    print("\n" + "="*80)
    print("EXTRACTED DATA SUMMARY")
    print("="*80)
    
    if epoch_data:
        print(f"\n📊 WITHIN-EPOCH PROGRESS DATA")
        print(f"   Epochs: {sorted(epoch_data.keys())}")
        for epoch in sorted(epoch_data.keys())[:5]:  # Show first 5
            ed = epoch_data[epoch]
            print(f"   Epoch {epoch}: {len(ed['steps'])} data points")
    
    if summaries:
        print(f"\n📈 EPOCH SUMMARIES")
        print(f"   Total epochs: {len(summaries)}")
        for summary in summaries[:5]:  # Show first 5
            print(f"   Epoch {summary['epoch']}: Train={summary['train_loss']:.4f}, Val={summary['val_loss']:.4f}")
        if len(summaries) > 5:
            print(f"   ... and {len(summaries) - 5} more epochs")
    
    print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract training data from raw Kaggle logs'
    )
    parser.add_argument('--log_file', type=str, required=True,
                        help='Raw Kaggle log file')
    parser.add_argument('--output', type=str, default='extracted_training_data.json',
                        help='Output JSON file')
    parser.add_argument('--visualize', action='store_true',
                        help='Also generate visualization graphs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXTRACTING KAGGLE TRAINING DATA")
    print("="*80)
    
    # Extract data
    data = extract_all_training_data(args.log_file)
    
    if not data['epoch_data'] and not data['summaries']:
        print("\n❌ No training data found in log file")
        return
    
    # Print summary
    print_summary(data)
    
    # Save to JSON
    output_file = save_extracted_data(data, args.output)
    
    # Optionally visualize
    if args.visualize:
        print("\n📊 Generating visualizations...")
        try:
            from parse_raw_kaggle_logs import plot_comprehensive_training_curves
            plot_comprehensive_training_curves(data, 'outputs')
        except ImportError as e:
            print(f"⚠️  Could not generate visualizations: {e}")
            print("   Run: python scripts/parse_raw_kaggle_logs.py --log_file extracted_training_data.json")
    
    print("\n✅ Done!")
    print(f"\nNext steps:")
    print(f"1. Review extracted data: {output_file}")
    print(f"2. Generate graphs: python scripts/parse_raw_kaggle_logs.py --log_file {args.log_file}")


if __name__ == "__main__":
    main()
