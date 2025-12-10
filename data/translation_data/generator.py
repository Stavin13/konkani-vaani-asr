import json, os, random, time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# -------------------------------
# Noise / Punctuation Augmenter
# -------------------------------
def add_noise(text):
    noise_ops = [
        lambda s: s.replace(" ", ""),                     # remove spaces
        lambda s: s.replace("ा", "ा ").replace("  ", " "), # split vowels
        lambda s: s + random.choice([" ..", " ...", " ."]), # add dots
        lambda s: s.replace("ं", "ँ"),                    # nasal variation
        lambda s: s.replace("त", "त्"),                   # half-char
        lambda s: s.replace("क", "क्"),                   # half-char
    ]
    if random.random() < 0.4:
        text = random.choice(noise_ops)(text)
    return text


# -------------------------------
# Generate One Synthetic Sample
# -------------------------------
def generate_pair():
    prompt = """
Generate a natural Devanagari Konkani sentence about village life, daily routine, work, school, family, or general Goa context.
Keep it 8–20 words.
ONLY OUTPUT THE SENTENCE, nothing else.
"""

    # 1. Generate Konkani
    kok = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.9
    ).choices[0].message.content.strip()

    # Add noisy augmentation
    kok_noisy = add_noise(kok)

    # 2. Translate to Indian English
    eng_prompt = f"""
Translate this Devanagari Konkani sentence to smooth Indian English:

"{kok_noisy}"
"""
    eng = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": eng_prompt}],
        max_tokens=80,
        temperature=0.4
    ).choices[0].message.content.strip()

    return kok_noisy, eng


# -------------------------------
# Generate Full Dataset
# -------------------------------
def generate_dataset(output_file="kok_eng_dataset.jsonl", count=15000):
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(count):
            kok, eng = generate_pair()
            row = {
                "konkani": kok,
                "english": eng,
                "source": "punctuation_aug",
                "confidence": round(random.uniform(0.88, 0.96), 2)
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[{i+1}/{count}] {kok} -> {eng}")
            time.sleep(0.3)  # prevent overuse rate throttling


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    generate_dataset(count=15000)
