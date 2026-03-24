import os
file_path = 'compare_beam.txt'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()
with open(file_path, 'rb') as f:
    data = f.read()
for enc in ['utf-16', 'utf-8', 'cp1252']:
    try:
        text = data.decode(enc)
        print(f"--- Decoded with {enc} ---")
        print(text)
        break
    except:
        continue
