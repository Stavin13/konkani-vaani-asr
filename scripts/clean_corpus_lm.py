import re
import os

def clean_and_truncate_corpus(input_file, output_file, max_lines=240251):
    print(f"Opening {input_file} for extraction...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find all occurrences of RECORDED TEXT :: followed by the content until TEXT TRANSLITERATION ::
    pattern = re.compile(r'RECORDED TEXT ::\s*(.*?)\s*TEXT TRANSLITERATION ::', re.DOTALL)
    matches = pattern.findall(content)

    clean_lines = []
    for match in matches:
        clean_text = match.strip()
        if clean_text:
            lines = [l.strip() for l in clean_text.split('\n') if l.strip()]
            for line in lines:
                clean_lines.append(line)

    print(f"Total extracted from {input_file}: {len(clean_lines)}")
    
    # Truncate at max_lines
    if len(clean_lines) > max_lines:
        print(f"Truncating to first {max_lines} lines...")
        clean_lines = clean_lines[:max_lines]

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in clean_lines:
            f.write(line + '\n')
            
    print(f"Successfully saved {len(clean_lines)} lines to {output_file}")

if __name__ == "__main__":
    input_path = r"E:\konkani\corpus_text.txt"
    # Target path from user's request
    output_path = r"E:\konkani\data\konkani_corpus_for_lm.txt"
    
    if os.path.exists(input_path):
        clean_and_truncate_corpus(input_path, output_path)
    else:
        print(f"Error: Source file {input_path} not found.")
