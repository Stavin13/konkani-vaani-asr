import pandas as pd
import os

stats_path = 'outputs/conformer_ctc_run1/training_stats.csv'
if os.path.exists(stats_path):
    df = pd.read_csv(stats_path)
    
    # Add missing columns if they don't exist
    if 'val_loss' not in df.columns:
        df['val_loss'] = 0.0
    if 'wer' not in df.columns:
        df['wer'] = 1.0 # Default to 100% error for visualization
    if 'cer' not in df.columns:
        df['cer'] = 1.0
        
    # Reorder columns to match new script
    final_cols = ['epoch', 'train_loss', 'val_loss', 'wer', 'cer', 'lr', 'timestamp']
    # If some cols like timestamp were missing or different, handle that
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0
            
    df = df[final_cols]
    df.to_csv(stats_path, index=False)
    print(f"Successfully migrated {stats_path} to the new 4-metric format.")
else:
    print("Stats file not found, no migration needed.")
