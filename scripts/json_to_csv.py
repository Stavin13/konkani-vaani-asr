import json
import csv
import os
from pathlib import Path

def convert_json_to_excel_csv(json_path, csv_path):
    """Converts our ASR comparison JSON into an Excel-friendly CSV with side-by-side strategy columns"""
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    details = data.get('detailed_predictions', {})
    strategies = list(details.keys())
    
    if not strategies:
        print("Error: No predictions found in JSON!")
        return

    # Total samples
    num_samples = len(details[strategies[0]])
    
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ['Sample #', 'Reference'] + [f'Pred ({s})' for s in strategies]
        writer.writerow(header)
        
        for i in range(num_samples):
            ref = details[strategies[0]][i]['reference']
            row = [i + 1, ref]
            for s in strategies:
                pred = details[s][i]['prediction']
                row.append(pred)
            writer.writerow(row)

    print(f"✅ Successfully converted {json_path} to Excel-ready {csv_path}")

if __name__ == "__main__":
    BASE = Path("/Volumes/data&proj/konkani")
    JSON_FILE = BASE / "outputs/beam_search_comparison.json"
    CSV_FILE = BASE / "outputs/beam_search_comparison.csv"
    
    convert_json_to_excel_csv(JSON_FILE, CSV_FILE)
