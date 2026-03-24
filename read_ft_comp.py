import os
p = 'ft_comparison.txt'
if os.path.exists(p):
    with open(p, 'rb') as f:
        data = f.read()
    for enc in ['utf-16', 'utf-8', 'cp1252']:
        try:
            print(f"--- {enc} ---")
            print(data.decode(enc))
            break
        except:
            continue
