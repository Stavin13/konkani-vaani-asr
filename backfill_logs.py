import csv
import os

stats_path = 'outputs/conformer_ctc_run1/training_stats.csv'
os.makedirs('outputs/conformer_ctc_run1', exist_ok=True)

# Data parsed from user logs
logs = [
    (3, 3.7938), (4, 3.5965), (5, 3.3192), (6, 2.9899), (7, 2.5965),
    (8, 2.2819), (9, 2.0524), (10, 1.8664), (11, 1.7527), (12, 1.6434),
    (13, 1.5454), (14, 1.4457), (15, 1.3327), (16, 1.2600), (17, 1.1704),
    (18, 1.0944), (19, 1.0325), (20, 0.9576), (21, 0.8932), (22, 0.8278),
    (23, 0.7688), (24, 0.7188), (25, 0.6694), (26, 0.6144), (27, 0.5720),
    (28, 0.5252)
]

# Write to CSV
with open(stats_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'lr', 'timestamp'])
    for epoch, loss in logs:
        # We don't have exact historical LR, using a placeholder for the graph
        writer.writerow([epoch, loss, 3.0e-4, '2026-02-27 18:44:00'])

print(f"Successfully backfilled {len(logs)} epochs into {stats_path}")
