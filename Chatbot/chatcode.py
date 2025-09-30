# Imports
# streamlit for UI, os for temporary files
import streamlit as st
import os
import tempfile
import logging
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_mic_recorder import mic_recorder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# list of response styles from GoEmotions based on detected emotion
response_styles = {
    "joy": "Celebrate with warmth and encouragement.",
    "sadness": "Respond gently, offering comfort and reassurance.",
    "anger": "Stay calm, validate feelings, and de-escalate.",
    "fear": "Reassure the user and provide a sense of safety.",
    "surprise": "Respond with curiosity and engagement.",
    "love": "Acknowledge kindness and care warmly.",
    "confusion": "Provide a simple, clear explanation.",
    "optimism": "Encourage positivity and motivation.",
    "neutral": "Respond in a straightforward and balanced way.",
    "other": "Offer a gentle, open-ended response."
}

# Initialize Streamlit page config
st.set_page_config(
    page_title="Senior Companion Chatbot", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Model loading functions with better error handling and optimization
@st.cache_resource(show_spinner="Loading your personal companion...")
def load_emotion_model():
    """Load emotion classification model with fallback options."""
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        device=-1,  # Force CPU usage
        return_all_scores=False
    )

@st.cache_resource(show_spinner="Loading your personal companion...")
def load_chatbot():
    """Load chatbot model with memory optimization."""
    try:
        model_name = "facebook/blenderbot-400M-distill"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load chatbot model: {e}")
        return None, None

@st.cache_resource(show_spinner="Loading speech recognition...")
def load_whisper():
    """Load Whisper model for speech-to-text."""
    try:
        return pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-small",
            device=-1,  # Force CPU usage
            chunk_length_s=30
        )
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None

# Initialize models with progress indicators
if 'models_loaded' not in st.session_state:
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        st.session_state.emotion_classifier = load_emotion_model()
        st.session_state.chat_tokenizer, st.session_state.chat_model = load_chatbot()
        st.session_state.whisper_asr = load_whisper()
        st.session_state.models_loaded = True
        st.success("✅ Models loaded successfully!")
else:
    # Use cached models
    emotion_classifier = st.session_state.emotion_classifier
    chat_tokenizer = st.session_state.chat_tokenizer
    chat_model = st.session_state.chat_model
    whisper_asr = st.session_state.whisper_asr

# Helper functions with improved error handling
def generate_reply(user_input: str, emotion: str) -> str:
    """Generate a contextual reply based on user input and detected emotion."""
    try:
        # Get the appropriate style instruction based on detected emotion
        style_instruction = response_styles.get(emotion.lower(), response_styles["other"])
        context_input = f"The user feels {emotion}. {style_instruction} User said: {user_input}"
        
        inputs = chat_tokenizer([context_input], return_tensors="pt", truncation=True, max_length=512)
        reply_ids = chat_model.generate(
            **inputs, 
            max_length=100, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=chat_tokenizer.eos_token_id
        )
        reply = chat_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Clean up the reply
        if "User said:" in reply:
            reply = reply.split("User said:")[-1].strip()
        
        # Fallback if reply is too short or empty
        if len(reply.strip()) < 3:
            reply = f"I understand you're feeling {emotion.lower()}. How can I help you today?"
            
        return reply

    except Exception as e:
        logger.error(f"Error generating reply: {e}")
        return f"I'm here to listen and help. You mentioned feeling {emotion.lower()}. Can you tell me more?"

def process_audio_file(uploaded_file) -> str:
    """Process audio and return transcription."""
    try:
        if is_uploaded and audio_data is not None:
            # Uploaded files saved to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data.read())
                tmp_file_path = tmp.name
            transcription = whisper_asr(audio_data.read())
            os.unlink(tmp_path)
    except Exception as e: 
        logger.error(f"Error processing file: {e}")
        return None 

# process audio with Whisper
def process_audio_bytes(audio_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        transcription = whisper_asr(tmp_path)
        os.unlink(tmp_path)
        return transcription.get("text", "").strip()
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

# ---------------------------------------------------- Main UI ---------------------------------
st.title("🧓 GoldenPal")
st.write("Providing companionship in your golden years, GoldenPal will respond with empathy and clear speech give your speech or text input.")

# Add a sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot uses AI to:")
    st.write("• 🎭 Detect emotions in your messages")
    st.write("• 💬 Respond with empathy and understanding")
    st.write("• 🎤 Live Voice Input")
    
    st.header("🔧 Status")
    if st.session_state.get('models_loaded', False):
        st.success("✅ All models loaded")
    else:
        st.error("❌ Models not loaded")

# Input methods
st.subheader("🎤 Live Voice Input")
audio_bytes = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    format="wav",
    just_once=True,
    key="recorder"
)

# 📁 Upload audio file
st.subheader("📁 Upload Audio File")
uploaded_audio = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'flac', 'm4a']
)

# ⌨️ Text input
st.subheader("⌨️ Text Input")
user_input = st.text_area(
    "Type your message here:",
    height=100,
    placeholder="Tell me how you're feeling or what's on your mind..."
)

# 🚀 Submit button
if st.button("🚀 Submit", type="primary", use_container_width=True):
    final_input = None

    if audio_bytes:
        st.info("🎤 Processing mic recording...")
        final_input = process_audio_bytes(audio_bytes)
        if final_input:
            st.write(f"**Transcribed (mic):** {final_input}")
        else:
            st.error("❌ Failed to transcribe mic recording.")

    elif uploaded_audio:
        st.info("📁 Processing uploaded file...")
        final_input = process_audio_file(uploaded_audio)
        if final_input:
            st.write(f"**Transcribed (file):** {final_input}")
        else:
            st.error("❌ Failed to transcribe uploaded audio.")

    elif user_input and user_input.strip():
        final_input = user_input.strip()
    else:
        st.warning("⚠️ Please provide input (mic, audio file, or text).")

    if final_input:
        with st.spinner("🤖 Analyzing your message..."):
            emotion_res = emotion_classifier(final_input, top_k=1)[0]
            detected_emotion = emotion_res.get("label", "other")
            confidence = emotion_res.get("score", 0.0)

            reply = generate_reply(final_input, detected_emotion)

        # Show emotion + reply
        col1, col2 = st.columns(2)
        col1.metric("Detected Emotion", detected_emotion.title())
        col2.metric("Confidence", f"{confidence:.1%}")

        st.markdown(
            f"""
            <div style='font-size:26px; color:#2E8B57; padding:20px; 
            background-color:#f0f8f0; border-radius:10px; 
            border-left: 5px solid #2E8B57;'>
            <strong>{reply}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )