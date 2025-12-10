#!/usr/bin/env python3
"""
Quick test: Does Whisper work on Konkani?
"""
import whisper
import sys
from pathlib import Path

def test_whisper_on_konkani():
    """Test Whisper on a Konkani audio sample"""
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Find a test audio file
    test_audio = None
    audio_paths = [
        "data/konkani-asr-v0/data/processed_segments_diarized/audio_segments/segment_000008.wav",
        "KonkaniRawSpeechCorpus/Data/Audio/konkani_001.wav"
    ]
    
    for path in audio_paths:
        if Path(path).exists():
            test_audio = path
            break
    
    if not test_audio:
        print("❌ No test audio found")
        return
    
    print(f"Testing on: {test_audio}")
    
    # Transcribe
    result = model.transcribe(test_audio, language="hi")  # Hindi as closest
    
    print("\nWhisper Result:")
    print(f"Text: {result['text']}")
    print(f"Language detected: {result.get('language', 'unknown')}")
    
    # Also try without language hint
    result_auto = model.transcribe(test_audio)
    print("\nWhisper (auto-detect):")
    print(f"Text: {result_auto['text']}")
    print(f"Language detected: {result_auto.get('language', 'unknown')}")

if __name__ == "__main__":
    test_whisper_on_konkani()
