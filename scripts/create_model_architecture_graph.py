#!/usr/bin/env python3
"""
Create Model Architecture Graphs for NER, Emotion, and Translation Models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_ner_model_graph():
    """Create NER model architecture graph"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Colors
    input_color = '#E8F4FD'
    embedding_color = '#B3E5FC'
    lstm_color = '#81C784'
    crf_color = '#FFB74D'
    output_color = '#F8BBD9'
    
    # Input layer
    input_box = FancyBboxPatch((1, 8.5), 3, 1, boxstyle="round,pad=0.1", 
                               facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 9, 'Konkani Text\n"हांव Mumbai वचता"', ha='center', va='center', fontsize=10, weight='bold')
    
    # Word tokenization
    word_box = FancyBboxPatch((0.5, 7), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=embedding_color, edgecolor='black')
    ax.add_patch(word_box)
    ax.text(1.5, 7.4, 'Word Tokens\n["हांव", "Mumbai", "वचता"]', ha='center', va='center', fontsize=9)
    
    # Character tokenization
    char_box = FancyBboxPatch((3, 7), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=embedding_color, edgecolor='black')
    ax.add_patch(char_box)
    ax.text(4, 7.4, 'Char Features\n[CNN on chars]', ha='center', va='center', fontsize=9)
    
    # Word embeddings
    word_emb_box = FancyBboxPatch((0.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=embedding_color, edgecolor='black')
    ax.add_patch(word_emb_box)
    ax.text(1.5, 5.9, 'Word Embeddings\n(vocab_size, 128)', ha='center', va='center', fontsize=9)
    
    # Character CNN
    char_cnn_box = FancyBboxPatch((3, 5.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=embedding_color, edgecolor='black')
    ax.add_patch(char_cnn_box)
    ax.text(4, 5.9, 'Char CNN\n(32→50 dims)', ha='center', va='center', fontsize=9)
    
    # Concatenation
    concat_box = FancyBboxPatch((1.5, 4), 2, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='#E1F5FE', edgecolor='black')
    ax.add_patch(concat_box)
    ax.text(2.5, 4.4, 'Concatenate\n(128 + 50 = 178)', ha='center', va='center', fontsize=9)
    
    # BiLSTM
    lstm_box = FancyBboxPatch((1, 2.5), 3, 1, boxstyle="round,pad=0.1", 
                              facecolor=lstm_color, edgecolor='black', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(2.5, 3, 'BiLSTM (2 layers)\nHidden: 256\nBidirectional', ha='center', va='center', fontsize=10, weight='bold')
    
    # Linear projection
    linear_box = FancyBboxPatch((1.5, 1), 2, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='#FFECB3', edgecolor='black')
    ax.add_patch(linear_box)
    ax.text(2.5, 1.4, 'Linear Layer\n(256 → 9 tags)', ha='center', va='center', fontsize=9)
    
    # CRF Layer
    crf_box = FancyBboxPatch((6, 2.5), 3, 1, boxstyle="round,pad=0.1", 
                             facecolor=crf_color, edgecolor='black', linewidth=2)
    ax.add_patch(crf_box)
    ax.text(7.5, 3, 'CRF Layer\nTransition Scores\nViterbi Decoding', ha='center', va='center', fontsize=10, weight='bold')
    
    # Output
    output_box = FancyBboxPatch((6.5, 0.5), 2, 1, boxstyle="round,pad=0.1", 
                                facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.5, 1, 'NER Tags\nO B-PER B-LOC\n(BIO format)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Tag legend
    tags_box = FancyBboxPatch((10, 7), 3, 2.5, boxstyle="round,pad=0.1", 
                              facecolor='#F5F5F5', edgecolor='black')
    ax.add_patch(tags_box)
    ax.text(11.5, 8.7, 'NER Tags', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(11.5, 8.2, 'O - Outside', ha='center', va='center', fontsize=9)
    ax.text(11.5, 7.9, 'B-PER - Begin Person', ha='center', va='center', fontsize=9)
    ax.text(11.5, 7.6, 'I-PER - Inside Person', ha='center', va='center', fontsize=9)
    ax.text(11.5, 7.3, 'B-ORG - Begin Organization', ha='center', va='center', fontsize=9)
    ax.text(11.5, 7.0, 'B-LOC - Begin Location', ha='center', va='center', fontsize=9)
    ax.text(11.5, 7.7, 'B-MISC - Begin Miscellaneous', ha='center', va='center', fontsize=9)
    
    # Arrows
    arrows = [
        ((2.5, 8.5), (2.5, 7.8)),  # Input to tokenization
        ((1.5, 7), (1.5, 6.3)),    # Word tokens to embeddings
        ((4, 7), (4, 6.3)),        # Char tokens to CNN
        ((1.5, 5.5), (2, 4.8)),    # Word emb to concat
        ((4, 5.5), (3, 4.8)),      # Char CNN to concat
        ((2.5, 4), (2.5, 3.5)),    # Concat to BiLSTM
        ((2.5, 2.5), (2.5, 1.8)),  # BiLSTM to linear
        ((3.5, 1.4), (6, 3)),      # Linear to CRF
        ((7.5, 2.5), (7.5, 1.5))   # CRF to output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 10)
    ax.set_title('Konkani NER Model Architecture\nBiLSTM-CRF with Word + Character Features', 
                fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_emotion_model_graph():
    """Create Emotion model architecture graph"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Colors
    input_color = '#E8F4FD'
    embedding_color = '#B3E5FC'
    lstm_color = '#81C784'
    attention_color = '#FFB74D'
    classifier_color = '#F8BBD9'
    
    # Input
    input_box = FancyBboxPatch((4, 9), 4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 9.4, 'Konkani Text: "हांव खुश आसा"', ha='center', va='center', fontsize=11, weight='bold')
    
    # Character tokenization
    char_box = FancyBboxPatch((4, 7.8), 4, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=embedding_color, edgecolor='black')
    ax.add_patch(char_box)
    ax.text(6, 8.2, 'Character Tokenization\n["ह", "ा", "ं", "व", " ", "ख", "ु", "श", ...]', ha='center', va='center', fontsize=10)
    
    # Character embeddings
    emb_box = FancyBboxPatch((4, 6.5), 4, 0.8, boxstyle="round,pad=0.1", 
                             facecolor=embedding_color, edgecolor='black')
    ax.add_patch(emb_box)
    ax.text(6, 6.9, 'Character Embeddings\n(vocab_size: ~5000, dim: 128)', ha='center', va='center', fontsize=10)
    
    # BiLSTM
    lstm_box = FancyBboxPatch((3.5, 5), 5, 1, boxstyle="round,pad=0.1", 
                              facecolor=lstm_color, edgecolor='black', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(6, 5.5, 'BiLSTM (2 layers)\nHidden dim: 256\nBidirectional + Dropout(0.3)', ha='center', va='center', fontsize=11, weight='bold')
    
    # Attention mechanism
    attention_box = FancyBboxPatch((3.5, 3.5), 5, 1, boxstyle="round,pad=0.1", 
                                   facecolor=attention_color, edgecolor='black', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(6, 4, 'Attention Layer\nCompute attention weights\nWeighted sum of LSTM outputs', ha='center', va='center', fontsize=11, weight='bold')
    
    # Classifier
    classifier_box = FancyBboxPatch((4, 2), 4, 1, boxstyle="round,pad=0.1", 
                                    facecolor=classifier_color, edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(6, 2.5, 'Classifier\nFC(256 → 7)\nSoftmax', ha='center', va='center', fontsize=11, weight='bold')
    
    # Output emotions
    output_box = FancyBboxPatch((3.5, 0.3), 5, 1, boxstyle="round,pad=0.1", 
                                facecolor='#E8F5E8', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 0.8, 'Emotion Prediction\n[joy: 0.85, neutral: 0.10, ...]', ha='center', va='center', fontsize=11, weight='bold')
    
    # Emotion classes legend
    emotions_box = FancyBboxPatch((9.5, 4), 2.5, 4, boxstyle="round,pad=0.1", 
                                  facecolor='#F5F5F5', edgecolor='black')
    ax.add_patch(emotions_box)
    ax.text(10.75, 7.5, 'Emotions', ha='center', va='center', fontsize=12, weight='bold')
    emotions = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral']
    for i, emotion in enumerate(emotions):
        ax.text(10.75, 7 - i*0.4, f'{i+1}. {emotion}', ha='center', va='center', fontsize=10)
    
    # Arrows
    arrows = [
        ((6, 9), (6, 8.6)),      # Input to tokenization
        ((6, 7.8), (6, 7.3)),    # Tokenization to embeddings
        ((6, 6.5), (6, 6)),      # Embeddings to BiLSTM
        ((6, 5), (6, 4.5)),      # BiLSTM to attention
        ((6, 3.5), (6, 3)),      # Attention to classifier
        ((6, 2), (6, 1.3))       # Classifier to output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 10)
    ax.set_title('Konkani Emotion Detection Model\nBiLSTM + Attention Architecture', 
                fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_translation_model_graph():
    """Create Translation model architecture graph"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Colors
    input_color = '#E8F4FD'
    nllb_color = '#81C784'
    custom_color = '#FFB74D'
    output_color = '#F8BBD9'
    
    # Title
    ax.text(8, 11.5, 'Konkani Translation Models', ha='center', va='center', 
            fontsize=18, weight='bold')
    
    # NLLB Model (Left side)
    ax.text(4, 10.8, 'NLLB-200 Model (Primary)', ha='center', va='center', 
            fontsize=14, weight='bold', color='darkgreen')
    
    # NLLB Input
    nllb_input = FancyBboxPatch((1, 9.5), 6, 0.8, boxstyle="round,pad=0.1", 
                                facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(nllb_input)
    ax.text(4, 9.9, 'Konkani Text: "हांव घरा वचता"', ha='center', va='center', fontsize=11, weight='bold')
    
    # NLLB Tokenizer
    tokenizer_box = FancyBboxPatch((1.5, 8.3), 5, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='#E3F2FD', edgecolor='black')
    ax.add_patch(tokenizer_box)
    ax.text(4, 8.7, 'SentencePiece Tokenizer\nLanguage: kok_Deva → eng_Latn', ha='center', va='center', fontsize=10)
    
    # NLLB Encoder
    encoder_box = FancyBboxPatch((1.5, 7), 5, 1.2, boxstyle="round,pad=0.1", 
                                 facecolor=nllb_color, edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(4, 7.6, 'NLLB Encoder\n24 Transformer Layers\n16 Attention Heads\n1024 Hidden Dim', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # NLLB Decoder
    decoder_box = FancyBboxPatch((1.5, 5.5), 5, 1.2, boxstyle="round,pad=0.1", 
                                 facecolor=nllb_color, edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(4, 6.1, 'NLLB Decoder\nAuto-regressive Generation\nBeam Search (beam=5)\nForced BOS: eng_Latn', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # NLLB Output
    nllb_output = FancyBboxPatch((1, 4), 6, 0.8, boxstyle="round,pad=0.1", 
                                 facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(nllb_output)
    ax.text(4, 4.4, 'English Translation: "I am going home"', ha='center', va='center', fontsize=11, weight='bold')
    
    # Custom Models (Right side)
    ax.text(12, 10.8, 'Custom Translation Models', ha='center', va='center', 
            fontsize=14, weight='bold', color='darkorange')
    
    # Custom Input
    custom_input = FancyBboxPatch((9, 9.5), 6, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(custom_input)
    ax.text(12, 9.9, 'Konkani Text: "हांव घरा वचता"', ha='center', va='center', fontsize=11, weight='bold')
    
    # Seq2Seq Model
    seq2seq_box = FancyBboxPatch((9, 8), 6, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor=custom_color, edgecolor='black', linewidth=2)
    ax.add_patch(seq2seq_box)
    ax.text(12, 8.75, 'Seq2Seq with Attention\nBiLSTM Encoder (2 layers)\nLSTM Decoder with Bahdanau Attention\nHidden: 512, Embedding: 256', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Transformer Model
    transformer_box = FancyBboxPatch((9, 6), 6, 1.5, boxstyle="round,pad=0.1", 
                                     facecolor=custom_color, edgecolor='black', linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(12, 6.75, 'Custom Transformer\n6 Encoder + 6 Decoder Layers\n8 Attention Heads\nd_model: 256, FFN: 1024', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Custom Output
    custom_output = FancyBboxPatch((9, 4), 6, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(custom_output)
    ax.text(12, 4.4, 'English Translation: "I am going home"', ha='center', va='center', fontsize=11, weight='bold')
    
    # Model comparison
    comparison_box = FancyBboxPatch((2, 1.5), 12, 2, boxstyle="round,pad=0.1", 
                                    facecolor='#F5F5F5', edgecolor='black', linewidth=2)
    ax.add_patch(comparison_box)
    ax.text(8, 3, 'Model Comparison', ha='center', va='center', fontsize=14, weight='bold')
    
    comparison_text = """
NLLB-200 (Primary):                          Custom Models (Experimental):
• Size: 2.4GB (600M parameters)              • Size: ~50-100MB each
• Speed: 0.5s per sentence                   • Speed: 0.2-0.3s per sentence  
• Quality: State-of-the-art                  • Quality: Good (with training data)
• Languages: 200+ supported                  • Languages: Konkani-English only
• Training: Pre-trained + Fine-tuned         • Training: From scratch on custom data
    """
    
    ax.text(8, 2.2, comparison_text, ha='center', va='center', fontsize=10, 
            fontfamily='monospace')
    
    # Arrows for NLLB
    nllb_arrows = [
        ((4, 9.5), (4, 9.1)),    # Input to tokenizer
        ((4, 8.3), (4, 8.2)),    # Tokenizer to encoder
        ((4, 7), (4, 6.7)),      # Encoder to decoder
        ((4, 5.5), (4, 4.8))     # Decoder to output
    ]
    
    for start, end in nllb_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    # Arrows for Custom
    custom_arrows = [
        ((12, 9.5), (12, 9.5)),   # Input to models
        ((12, 8), (12, 7.5)),     # Between models
        ((12, 6), (12, 4.8))      # Models to output
    ]
    
    for start, end in custom_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkorange'))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_combined_pipeline_graph():
    """Create combined pipeline showing all three models"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # Colors
    input_color = '#E8F4FD'
    ner_color = '#81C784'
    emotion_color = '#FFB74D'
    translation_color = '#F8BBD9'
    output_color = '#E8F5E8'
    
    # Title
    ax.text(9, 11.5, 'Konkani NLP Pipeline: NER + Emotion + Translation', 
            ha='center', va='center', fontsize=18, weight='bold')
    
    # Input
    input_box = FancyBboxPatch((7, 10), 4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=input_color, edgecolor='black', linewidth=3)
    ax.add_patch(input_box)
    ax.text(9, 10.4, 'Konkani Text Input\n"हांव Mumbai ला खुश आसा"', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # NER Branch
    ner_box = FancyBboxPatch((1, 7.5), 4, 2, boxstyle="round,pad=0.1", 
                             facecolor=ner_color, edgecolor='black', linewidth=2)
    ax.add_patch(ner_box)
    ax.text(3, 8.5, 'NER Model\nBiLSTM-CRF\n\nOutput:\n• "Mumbai" → B-LOC\n• "हांव" → O', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Emotion Branch
    emotion_box = FancyBboxPatch((7, 7.5), 4, 2, boxstyle="round,pad=0.1", 
                                 facecolor=emotion_color, edgecolor='black', linewidth=2)
    ax.add_patch(emotion_box)
    ax.text(9, 8.5, 'Emotion Model\nBiLSTM + Attention\n\nOutput:\n• Joy: 0.75\n• Neutral: 0.20', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Translation Branch
    translation_box = FancyBboxPatch((13, 7.5), 4, 2, boxstyle="round,pad=0.1", 
                                     facecolor=translation_color, edgecolor='black', linewidth=2)
    ax.add_patch(translation_box)
    ax.text(15, 8.5, 'Translation Model\nNLLB-200\n\nOutput:\n"I am happy\nto go to Mumbai"', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Combined Output
    output_box = FancyBboxPatch((4, 4), 10, 2.5, boxstyle="round,pad=0.1", 
                                facecolor=output_color, edgecolor='black', linewidth=3)
    ax.add_patch(output_box)
    
    output_text = """Structured JSON Output:
{
  "original_text": "हांव Mumbai ला खुश आसा",
  "english_translation": "I am happy to go to Mumbai",
  "emotion": {"label": "joy", "confidence": 0.75},
  "entities": [{"text": "Mumbai", "type": "LOC", "start": 1, "end": 1}]
}"""
    
    ax.text(9, 5.25, output_text, ha='center', va='center', fontsize=11, 
            weight='bold', fontfamily='monospace')
    
    # Model details boxes
    details_y = 1.5
    
    # NER details
    ner_details = FancyBboxPatch((0.5, details_y), 5, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor='#E8F5E8', edgecolor='black')
    ax.add_patch(ner_details)
    ax.text(3, details_y + 0.75, 'NER Model Details:\n• Vocab: ~5000 words\n• Tags: 9 (BIO format)\n• Architecture: BiLSTM-CRF\n• Features: Word + Char embeddings', 
            ha='center', va='center', fontsize=9)
    
    # Emotion details
    emotion_details = FancyBboxPatch((6.5, details_y), 5, 1.5, boxstyle="round,pad=0.1", 
                                     facecolor='#FFF3E0', edgecolor='black')
    ax.add_patch(emotion_details)
    ax.text(9, details_y + 0.75, 'Emotion Model Details:\n• Classes: 7 emotions\n• Input: Character-level\n• Architecture: BiLSTM + Attention\n• Size: ~10MB', 
            ha='center', va='center', fontsize=9)
    
    # Translation details
    translation_details = FancyBboxPatch((12.5, details_y), 5, 1.5, boxstyle="round,pad=0.1", 
                                         facecolor='#FCE4EC', edgecolor='black')
    ax.add_patch(translation_details)
    ax.text(15, details_y + 0.75, 'Translation Model Details:\n• Model: NLLB-200 (600M)\n• Languages: kok_Deva ↔ eng_Latn\n• Method: Transformer seq2seq\n• Size: 2.4GB', 
            ha='center', va='center', fontsize=9)
    
    # Arrows
    arrows = [
        # From input to models
        ((8, 10), (3, 9.5)),     # Input to NER
        ((9, 10), (9, 9.5)),     # Input to Emotion
        ((10, 10), (15, 9.5)),   # Input to Translation
        
        # From models to output
        ((3, 7.5), (6, 6.5)),    # NER to output
        ((9, 7.5), (9, 6.5)),    # Emotion to output
        ((15, 7.5), (12, 6.5))   # Translation to output
    ]
    
    colors = ['green', 'orange', 'purple', 'green', 'orange', 'purple']
    
    for i, (start, end) in enumerate(arrows):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors[i]))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all model architecture graphs"""
    print("Creating model architecture graphs...")
    
    # Create output directory
    import os
    os.makedirs('outputs/model_graphs', exist_ok=True)
    
    # Generate individual model graphs
    print("1. Creating NER model graph...")
    ner_fig = create_ner_model_graph()
    ner_fig.savefig('outputs/model_graphs/ner_model_architecture.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(ner_fig)
    
    print("2. Creating Emotion model graph...")
    emotion_fig = create_emotion_model_graph()
    emotion_fig.savefig('outputs/model_graphs/emotion_model_architecture.png', 
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(emotion_fig)
    
    print("3. Creating Translation model graph...")
    translation_fig = create_translation_model_graph()
    translation_fig.savefig('outputs/model_graphs/translation_model_architecture.png', 
                            dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(translation_fig)
    
    print("4. Creating combined pipeline graph...")
    pipeline_fig = create_combined_pipeline_graph()
    pipeline_fig.savefig('outputs/model_graphs/combined_pipeline_architecture.png', 
                         dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(pipeline_fig)
    
    print("\n✅ All graphs created successfully!")
    print("📁 Saved to: outputs/model_graphs/")
    print("   - ner_model_architecture.png")
    print("   - emotion_model_architecture.png") 
    print("   - translation_model_architecture.png")
    print("   - combined_pipeline_architecture.png")

if __name__ == "__main__":
    main()