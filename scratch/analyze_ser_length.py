import openpyxl
import os

file_path = "outputs/predictions_analysis.xlsx"
if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
    exit(1)

wb = openpyxl.load_workbook(file_path)
# Check all sheets for a 'Detailed Predictions' one
sheet_names = wb.sheetnames
ws = wb.active
if "Detailed Predictions" in sheet_names:
    ws = wb["Detailed Predictions"]

ser0_len = []
ser1_len = []

# Headers are likely: Audio, Reference, 3-gram Hyp, S3, D3, I3, 4-gram Hyp, S4, D4, I4
# In the April 14 version, it might be different. Let's find the 'Reference' and '3-gram' columns.
rows = list(ws.rows)
header = [str(cell.value) for cell in rows[0]]
ref_idx = -1
hyp_idx = -1

for i, h in enumerate(header):
    if "Reference" in h: ref_idx = i
    if "3-gram" in h or "Hyp" in h or "hyp_lm" in h: 
        if hyp_idx == -1: hyp_idx = i

if ref_idx == -1 or hyp_idx == -1:
    print(f"Could not find Reference or Hyp columns. Header: {header}")
    exit(1)

for row in rows[1:]:
    ref = str(row[ref_idx].value) if row[ref_idx].value else ""
    hyp = str(row[hyp_idx].value) if row[hyp_idx].value else ""
    
    # We'll use 4-gram for the current stats if available, else 3-gram
    # Let's check 4-gram too
    hyp4_idx = -1
    for i, h in enumerate(header):
        if "4-gram" in h: hyp4_idx = i
    
    if hyp4_idx != -1:
        hyp = str(row[hyp4_idx].value) if row[hyp4_idx].value else ""

    length = len(ref.split())
    if length == 0: continue
    
    if ref.strip() == hyp.strip():
        ser0_len.append(length)
    else:
        ser1_len.append(length)

if ser0_len:
    print(f"SER=0 (Correct)   - Count: {len(ser0_len)}, Avg Words: {sum(ser0_len)/len(ser0_len):.2f}")
if ser1_len:
    print(f"SER=1 (Incorrect) - Count: {len(ser1_len)}, Avg Words: {sum(ser1_len)/len(ser1_len):.2f}")
