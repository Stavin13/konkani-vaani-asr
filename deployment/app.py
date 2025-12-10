"""
Streamlit Frontend for Konkani NLP Pipeline
"""
import streamlit as st
import tempfile
import sys
from pathlib import Path
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from pipeline import KonkaniPipeline

# Page config
st.set_page_config(
    page_title="KonkaniVani - Complete NLP Pipeline",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .entity-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .entity-per { background-color: #ffcccc; }
    .entity-org { background-color: #ccffcc; }
    .entity-loc { background-color: #ccccff; }
    .entity-misc { background-color: #ffffcc; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.results = None


@st.cache_resource
def load_pipeline(device):
    """Load pipeline (cached)"""
    return KonkaniPipeline(device=device)


def display_emotion_results(emotion_data):
    """Display emotion detection results"""
    st.markdown('<div class="sub-header">😊 Emotion Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Detected Emotion", emotion_data['label'].upper())
        st.metric("Confidence", f"{emotion_data['confidence']:.1%}")
    
    with col2:
        st.markdown("**All Emotion Scores:**")
        scores = emotion_data['all_scores']
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, score in sorted_scores:
            st.progress(score, text=f"{emotion}: {score:.1%}")


def display_entities(entities):
    """Display NER results"""
    st.markdown('<div class="sub-header">🏷️ Named Entities</div>', unsafe_allow_html=True)
    
    if not entities:
        st.info("No named entities detected")
        return
    
    # Group by type
    entities_by_type = {}
    for entity_text, entity_type, start, end in entities:
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity_text)
    
    # Display
    for entity_type, entity_list in entities_by_type.items():
        color_class = f"entity-{entity_type.lower()}"
        st.markdown(f"**{entity_type}:**")
        for entity in entity_list:
            st.markdown(
                f'<span class="entity-tag {color_class}">{entity}</span>',
                unsafe_allow_html=True
            )


def main():
    # Header
    st.markdown('<div class="main-header">🎤 KonkaniVani NLP Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
    Complete Konkani language processing: Speech Recognition • Translation • Emotion Detection • Named Entity Recognition
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device_options = ['cpu']
        if st.session_state.get('mps_available', False):
            device_options.insert(0, 'mps')
        if st.session_state.get('cuda_available', False):
            device_options.insert(0, 'cuda')
        
        device = st.selectbox("Device", device_options, index=0)
        
        # Processing options
        st.subheader("Processing Options")
        include_translation = st.checkbox("Translation", value=True)
        include_emotion = st.checkbox("Emotion Detection", value=True)
        include_ner = st.checkbox("Named Entity Recognition", value=True)
        
        # Load pipeline button
        if st.button("🚀 Initialize Pipeline", type="primary"):
            with st.spinner("Loading models..."):
                try:
                    st.session_state.pipeline = load_pipeline(device)
                    st.success("✅ Pipeline loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading pipeline: {e}")
    
    # Main content
    if st.session_state.pipeline is None:
        st.info("👈 Click 'Initialize Pipeline' in the sidebar to start")
        return
    
    # Tabs for different input modes
    tab1, tab2 = st.tabs(["🎤 Audio Input", "✍️ Text Input"])
    
    with tab1:
        st.markdown("### Upload Konkani Audio")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a Konkani audio file for transcription and analysis"
        )
        
        if audio_file:
            st.audio(audio_file)
            
            if st.button("🎯 Process Audio", type="primary"):
                with st.spinner("Processing audio..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
                        tmp_file.write(audio_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process
                        results = st.session_state.pipeline.process_audio(
                            tmp_path,
                            include_translation=include_translation,
                            include_emotion=include_emotion,
                            include_ner=include_ner
                        )
                        st.session_state.results = results
                        
                        # Clean up
                        Path(tmp_path).unlink()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {e}")
                        Path(tmp_path).unlink()
    
    with tab2:
        st.markdown("### Enter Konkani Text")
        konkani_text = st.text_area(
            "Konkani Text",
            height=150,
            placeholder="Enter Konkani text here...",
            help="Enter text in Konkani (Devanagari script)"
        )
        
        if st.button("🎯 Process Text", type="primary"):
            if konkani_text.strip():
                with st.spinner("Processing text..."):
                    try:
                        results = st.session_state.pipeline.process_text(
                            konkani_text,
                            include_translation=include_translation,
                            include_emotion=include_emotion,
                            include_ner=include_ner
                        )
                        st.session_state.results = results
                    except Exception as e:
                        st.error(f"❌ Error processing text: {e}")
            else:
                st.warning("Please enter some text")
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Results</div>', unsafe_allow_html=True)
        
        results = st.session_state.results
        
        # Transcription
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🇮🇳 Konkani Text:**")
            st.markdown(f'<div class="result-box" style="font-size: 1.2rem;">{results["konkani_text"]}</div>', 
                       unsafe_allow_html=True)
        
        with col2:
            if 'english_text' in results:
                st.markdown("**🇬🇧 English Translation:**")
                st.markdown(f'<div class="result-box" style="font-size: 1.2rem;">{results["english_text"]}</div>', 
                           unsafe_allow_html=True)
        
        # Emotion
        if 'emotion' in results:
            display_emotion_results(results['emotion'])
        
        # Entities
        if 'entities' in results:
            display_entities(results['entities'])
        
        # Download results
        st.markdown("---")
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(results, indent=2, ensure_ascii=False),
            file_name="konkani_nlp_results.json",
            mime="application/json"
        )


if __name__ == '__main__':
    main()
