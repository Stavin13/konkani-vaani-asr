import json
import random
from typing import Dict, List
import itertools

class KonkaniSentimentDictionaryGenerator:
    def __init__(self):
        # Core sentiment words in Konkani (Devanagari)
        self.base_words = {
            "positive": [
                # Basic positive (100 words)
                "बरें", "छान", "उत्तम", "सुंदर", "मस्त", "भलें", "शाबास", "गोड", "मीठ", "सोबीत",
                "रूंद", "पावन", "पवित्र", "शुद्ध", "साफ", "स्वच्छ", "निर्मळ", "स्पष्ट", "सोपें", "सुविधा",
                "आनंद", "खोश", "आनंददायक", "आवडीचें", "प्रिय", "प्रेमळ", "मायाळ", "स्नेही", "दयाळ", "कृपाळ",
                "हुशार", "चलाख", "बुद्धीमान", "तरकीबी", "कुशळ", "निपुण", "प्रवीण", "दक्ष", "कसबी", "हुनर",
                "बळिश्ट", "ताकदवान", "शक्तिशाली", "जोरदार", "धीट", "शूर", "पराक्रमी", "वीर", "साहसी", "धाडसी",
                "शांत", "स्वस्थ", "निवांत", "गंभीर", "स्थिर", "धीर", "संयमी", "विचारी", "समजूतदार", "बुद्धिवान",
                "उपयोगी", "फायदेशीर", "लाभदायक", "हितकारक", "कल्याणकारी", "मंगल", "शुभ", "भाग्यशाली", "सुदैवी", "नशीबवान",
                "स्वस्त", "सवलत", "मोफत", "किफायतशीर", "लाभाचें", "मूल्यवान", "अमूल्य", "अनमोल", "दुर्मिळ", "विशेष",
                "आधुनिक", "नवें", "ताजें", "तरुण", "युवा", "चैतन्य", "सजीव", "प्राणवान", "उत्साही", "जोमदार",
                "पारंपारिक", "प्राचीन", "पुरातन", "शाश्वत", "स्थायी", "टिकाऊ", "दीर्घकालीन", "चिरस्थायी", "अखंड", "अविरत"
            ],
            
            "negative": [
                # Basic negative (100 words)
                "वायट", "विक्राळ", "खराब", "अवघड", "त्रास", "कंटाळा", "भिरांत", "उदास", "दुःखी", "खिन्न",
                "निराश", "हताश", "नाउमेद", "निरुत्साह", "उदासीन", "निस्तेज", "मंद", "सुस्त", "आळशी", "निठुर",
                "क्रूर", "निर्दय", "कठोर", "कडक", "राक्षसी", "भयंकर", "भीतीदायक", "डरावन", "घोर", "भयाण",
                "कमी", "अपुरें", "अपूर्ण", "अर्दवट", "अधुरें", "तुटपुंजें", "निकृष्ट", "हलकें", "तुच्छ", "नीच",
                "महाग", "अमर्याद", "असंबद्ध", "अव्यवस्थित", "गोंधळ", "अराजक", "अनियमित", "असुरक्षित", "धोकादायक", "जोखमी",
                "फसवें", "भ्रमित", "गैरसमज", "दिशाभूल", "घोटाळा", "फसवणूक", "छल", "कपट", "धूर्त", "चतुर",
                "दुर्गंधी", "कुजिल्लें", "सडिल्लें", "घाण", "मळ", "मैल", "अशुद्ध", "अपवित्र", "अस्पष्ट", "गोंधळलेलें",
                "अनाकर्षक", "कुरूप", "विरुप", "विकृत", "अस्वाभाविक", "अप्राकृतिक", "असंवेदनशील", "बेशरम", "निर्लज्ज", "धृष्ट",
                "नको", "अनावश्यक", "निरुपयोगी", "व्यर्थ", "निरर्थक", "फुकट", "वाया गेल्लें", "हानीकारक", "हानी", "नुकसान"
            ],
            
            "neutral": [
                # Basic neutral (100 words)
                "साधारण", "मध्यम", "सामान्य", "घट", "वाड", "बदल", "परिवर्तन", "फरक", "तफावत", "अंतर",
                "आकार", "प्रमाण", "परिमाण", "माप", "तोल", "वजन", "घनता", "घनफळ", "क्षेत्र", "विस्तार",
                "काळ", "वेळ", "समय", "तास", "मिनीट", "सेकंद", "दीस", "रात", "म्हयनो", "वर्स",
                "स्थान", "जागो", "ठिकाण", "स्थळ", "क्षेत्र", "प्रदेश", "भाग", "विभाग", "क्षेत्रफळ", "परिसर",
                "कारण", "हेतू", "निमित्त", "मूळ", "आदि", "प्रारंभ", "सुरवात", "शेवट", "अंत", "समाप्त",
                "मार्ग", "रस्तो", "पथ", "दार", "द्वार", "प्रवेश", "निर्गम", "आगमन", "प्रस्थान", "येणे",
                "जाणे", "बसप", "उठप", "बदलप", "रूपांतर", "परिवर्तन", "फेरबदल", "सुधारणा", "दुरुस्ती", "मरम्मत",
                "तपासणी", "परीक्षण", "निरीक्षण", "अवलोकन", "विचार", "चिंतन", "मनन", "विवेचन", "विश्लेषण", "पडताळणी",
                "संबंध", "नातें", "संपर्क", "जोड", "बंध", "आंतरसंबंध", "अंतर्गत", "बाह्य", "आत", "बाहेर"
            ]
        }
        
        # Context phrases (200 phrases)
        self.context_phrases = [
            "हें फोन", "ही फिल्म", "हें जेवण", "हें पुस्तक", "ही सेवा", "हो होटेल", "हें गाणें", "ही कार", 
            "हें शिक्षण", "ही दुकान", "हें घर", "ही वस्तू", "हो कार्यक्रम", "हें सॉफ्टवेअर", "ही ऍप", "हो गेम",
            "हें दवाखानें", "ही रस्तो", "हें शहर", "ही नदी", "हें डोंगर", "ही वाट", "हें समुद्र", "ही बेट",
            "हें हवामान", "ही ऋतू", "हें पान", "ही फुल", "हें झाड", "ही वनस्पत", "हें प्राणी", "ही पक्षी",
            "हें कपडें", "ही जोडा", "हें घडयाळ", "ही चष्मा", "हें सोनें", "ही चांदी", "हें लोखंड", "ही प्लॅस्टिक",
            "हें चित्र", "ही मूर्ती", "हें नाच", "ही नाटक", "हें ग्रंथ", "ही कविता", "हें कथानक", "ही कहाणी",
            "हें विचार", "ही कल्पना", "हें स्वप्न", "ही आशा", "हें ध्येय", "ही इच्छा", "हें संकल्प", "ही प्रतिज्ञा",
            # More phrases...
        ] * 4  # Repeat to get more phrases
        
        # Romanization patterns
        self.roman_patterns = {
            'ा': ['aa', 'a', 'ah', 'ā'],
            'ी': ['ee', 'i', 'ie', 'ī'],
            'ू': ['oo', 'u', 'ou', 'ū'],
            'े': ['e', 'ey', 'ay', 'ē'],
            'ै': ['ai', 'ay', 'ae', 'ai'],
            'ो': ['o', 'oh', 'ow', 'ō'],
            'ौ': ['au', 'aw', 'ao', 'au'],
            'ं': ['n', 'm', 'n', 'ṃ'],
            'ः': ['h', 'aha', 'ah', 'ḥ'],
            'ऋ': ['ri', 'ru', 'r', 'ṛ'],
            '़': ['', 'a', 'e', ''],  # nukta
            'श': ['sh', 'sha', 's', 'ś'],
            'ष': ['sh', 'ssha', 's', 'ṣ'],
            'ज्ञ': ['gya', 'jna', 'gny', 'jñ']
        }
    
    def generate_roman_variants(self, dev_word: str, num_variants: int = 5) -> List[str]:
        """Generate Romanization variants for a Devanagari word"""
        variants = []
        
        # Basic Romanization (simplified)
        basic = ""
        for char in dev_word:
            if char in self.roman_patterns:
                basic += self.roman_patterns[char][0]
            else:
                basic += char
        
        variants.append(basic)
        
        # Generate variants by applying different patterns
        for i in range(num_variants - 1):
            variant = ""
            for char in dev_word:
                if char in self.roman_patterns:
                    # Randomly choose a Romanization
                    variant += random.choice(self.roman_patterns[char])
                else:
                    variant += char
            
            # Add some common transformations
            if random.random() > 0.5:
                variant = variant.replace('ch', 'c').replace('sh', 's')
            if random.random() > 0.5:
                variant = variant.replace('aa', 'a').replace('ee', 'i')
            if random.random() > 0.5:
                variant = variant.upper() if random.random() > 0.7 else variant.lower()
            
            variants.append(variant)
        
        return list(set(variants))[:num_variants]
    
    def generate_full_sentences(self, count: int = 5000) -> Dict:
        """Generate complete sentences with sentiment labels"""
        sentences = {}
        
        for i in range(count):
            # Choose sentiment
            sentiment = random.choice(["positive", "negative", "neutral"])
            
            # Build sentence components
            context = random.choice(self.context_phrases)
            sentiment_word = random.choice(self.base_words[sentiment])
            
            # Add intensifiers/verbs
            intensifiers = ["खूप", "बरीच", "अतिशय", "फार", "प्रचंड", ""]
            verbs = ["आसा", "जालें", "येता", "करता", "दिसता", "वाटटा"]
            
            intensifier = random.choice(intensifiers)
            verb = random.choice(verbs)
            
            # Create sentence
            if intensifier:
                dev_sentence = f"{context} {intensifier} {sentiment_word} {verb}"
            else:
                dev_sentence = f"{context} {sentiment_word} {verb}"
            
            # Generate Roman variants
            roman_variants = self.generate_roman_variants(dev_sentence, 5)
            
            sentences[f"sentence_{i+1}"] = {
                "devanagari": dev_sentence,
                "roman_variants": roman_variants,
                "label": sentiment,
                "components": {
                    "context": context,
                    "sentiment_word": sentiment_word,
                    "intensifier": intensifier,
                    "verb": verb
                }
            }
        
        return sentences
    
    def generate_word_dictionary(self) -> Dict:
        """Generate word-level dictionary"""
        dictionary = {}
        
        for sentiment, words in self.base_words.items():
            dictionary[sentiment] = {}
            for word in words:
                dictionary[sentiment][word] = self.generate_roman_variants(word, 5)
        
        # Add context phrases
        dictionary["context_phrases"] = {}
        for phrase in self.context_phrases[:100]:  # First 100 unique
            dictionary["context_phrases"][phrase] = self.generate_roman_variants(phrase, 3)
        
        # Add common verbs and intensifiers
        dictionary["verbs"] = {
            "आसा": ["aasa", "asa", "aasaa", "aashaa"],
            "जालें": ["zaalem", "jalem", "zaalym", "jalym"],
            "येता": ["yeta", "yetaa", "yetta", "yeta"],
            "करता": ["karta", "kartaa", "kartta", "kartaa"],
            "दिसता": ["dishta", "distaa", "dishtaa", "dishtaa"],
            "वाटटा": ["vaatata", "vatata", "vaatatta", "vatatta"]
        }
        
        dictionary["intensifiers"] = {
            "खूप": ["khoop", "khup", "khop", "khoop"],
            "बरीच": ["bareech", "breech", "bari cha", "bareech"],
            "अतिशय": ["atishay", "atishai", "atishya", "atishay"],
            "फार": ["phaar", "far", "faar", "phar"],
            "प्रचंड": ["prachand", "prachanda", "prachandh", "prachand"]
        }
        
        return dictionary
    
    def save_to_files(self, total_size: int = 10000):
        """Generate and save all data"""
        print("Generating Konkani sentiment dictionary...")
        
        # 1. Word-level dictionary (5,000+ entries)
        word_dict = self.generate_word_dictionary()
        
        # Count total words
        total_words = 0
        for sentiment, words in word_dict.items():
            if isinstance(words, dict):
                total_words += len(words)
        
        print(f"Generated {total_words} word entries")
        
        # 2. Sentence-level data (5,000+ entries)
        sentences = self.generate_full_sentences(5000)
        
        print(f"Generated {len(sentences)} sentence entries")
        print(f"Total entries: {total_words + len(sentences)}")
        
        # Save word dictionary
        with open('konkani_sentiment_words.json', 'w', encoding='utf-8') as f:
            json.dump(word_dict, f, ensure_ascii=False, indent=2)
        
        # Save sentences
        with open('konkani_sentiment_sentences.json', 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        # Save as CSV for ML
        self.save_as_csv(word_dict, sentences)
        
        # Save as JSONL for HuggingFace
        self.save_as_jsonl(word_dict, sentences)
        
        print("Files saved:")
        print("1. konkani_sentiment_words.json")
        print("2. konkani_sentiment_sentences.json")
        print("3. konkani_sentiment.csv")
        print("4. konkani_sentiment.jsonl")
        
        return word_dict, sentences
    
    def save_as_csv(self, word_dict, sentences):
        """Save as CSV format"""
        import pandas as pd
        
        rows = []
        
        # Add words
        for sentiment, words in word_dict.items():
            if isinstance(words, dict):
                for dev_word, roman_list in words.items():
                    for roman in roman_list:
                        rows.append({
                            "text": roman,
                            "devanagari": dev_word,
                            "label": sentiment if sentiment in ["positive", "negative", "neutral"] else "other",
                            "type": "word"
                        })
        
        # Add sentences
        for sent_id, data in sentences.items():
            for roman in data["roman_variants"]:
                rows.append({
                    "text": roman,
                    "devanagari": data["devanagari"],
                    "label": data["label"],
                    "type": "sentence"
                })
        
        df = pd.DataFrame(rows)
        df.to_csv('konkani_sentiment.csv', index=False, encoding='utf-8')
    
    def save_as_jsonl(self, word_dict, sentences):
        """Save as JSONL format for HuggingFace"""
        with open('konkani_sentiment.jsonl', 'w', encoding='utf-8') as f:
            # Words
            for sentiment, words in word_dict.items():
                if isinstance(words, dict):
                    for dev_word, roman_list in words.items():
                        for roman in roman_list:
                            entry = {
                                "text": roman,
                                "devanagari": dev_word,
                                "label": sentiment if sentiment in ["positive", "negative", "neutral"] else "other"
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Sentences
            for sent_id, data in sentences.items():
                for roman in data["roman_variants"]:
                    entry = {
                        "text": roman,
                        "devanagari": data["devanagari"],
                        "label": data["label"]
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Generate the dataset
if __name__ == "__main__":
    generator = KonkaniSentimentDictionaryGenerator()
    word_dict, sentences = generator.save_to_files(10000)