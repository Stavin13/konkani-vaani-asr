import os
with open('compare_results.txt', 'rb') as f:
    data = f.read()
# Try decoding as utf-16 with fallback to utf-8
for enc in ['utf-16', 'utf-8', 'cp1252']:
    try:
        text = data.decode(enc)
        print(f"--- Decoded with {enc} ---")
        print(text)
        break
    except:
        continue
