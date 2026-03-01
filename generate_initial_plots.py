import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
from train_conformer_ctc import save_plots

stats_path = 'outputs/conformer_ctc_run1/training_stats.csv'
output_dir = 'outputs/conformer_ctc_run1'

if os.path.exists(stats_path):
    save_plots(stats_path, output_dir)
    print(f"Successfully generated initial plots in {output_dir}/training_progress.png")
else:
    print(f"Stats file not found: {stats_path}")
