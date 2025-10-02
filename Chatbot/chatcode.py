# Imports
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

# Expanded emotion mapping with more nuanced responses
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
    "disgust": "Acknowledge concern and offer understanding.",
    "admiration": "Share in the appreciation warmly.",
    "disappointment": "Offer understanding and gentle encouragement.",
    "curiosity": "Engage with interest and provide helpful information.",
    "other": "Offer a gentle, open-ended response."
}

# Initialize browser tab and layout
st.set_page_config(
    page_title="Senior Companion Chatbot", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Improved emotion detection with ensemble approach
@st.cache_resource(show_spinner="Loading emotion detection...")
def load_emotion_model():
    """Load emotion classification model - using GoEmotions for better coverage."""
    try:
        # This model is specifically trained on GoEmotions dataset with 28 emotions
        return pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            device=-1,
            top_k=None  # Return all scores for better analysis
        )
    except Exception as e:
        logger.error(f"Failed to load GoEmotions model, falling back: {e}")
        # Fallback to original model
        return pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            device=-1,
            return_all_scores=True
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
            device=-1,
            chunk_length_s=30
        )
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None

# Initialize models
if 'models_loaded' not in st.session_state:
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        st.session_state.emotion_classifier = load_emotion_model()
        st.session_state.chat_tokenizer, st.session_state.chat_model = load_chatbot()
        st.session_state.whisper_asr = load_whisper()
        st.session_state.models_loaded = True
        st.success("✅ Models loaded successfully!")

emotion_classifier = st.session_state.emotion_classifier
chat_tokenizer = st.session_state.chat_tokenizer
chat_model = st.session_state.chat_model
whisper_asr = st.session_state.whisper_asr

def detect_emotion_advanced(text: str) -> tuple:
    """
    Advanced emotion detection with better handling of neutral and ambiguous statements.
    Returns: (primary_emotion, confidence, all_emotions_dict)
    """
    try:
        # Get all emotion scores
        results = emotion_classifier(text)
        
        # Convert to dictionary for easier processing
        emotion_scores = {item['label']: item['score'] for item in results}
        
        # Get top emotion
        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion = top_emotion[0]
        confidence = top_emotion[1]
        
        # Special handling for low-confidence predictions
        # If confidence is low and emotions are distributed, classify as neutral or confusion
        if confidence < 0.4:
            # Check if input contains question words
            question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', '?']
            if any(word in text.lower() for word in question_words):
                primary_emotion = "curiosity"
            else:
                primary_emotion = "neutral"
        
        # Detect confusion patterns
        confusion_indicators = ['confused', 'don\'t understand', 'not sure', 'unclear', 
                               'what do you mean', 'i don\'t get', 'huh', 'what']
        if any(indicator in text.lower() for indicator in confusion_indicators):
            primary_emotion = "confusion"
            confidence = max(confidence, 0.7)  # Boost confidence for clear confusion markers
        
        # Detect neutral statements (factual, informational)
        neutral_patterns = ['the weather', 'today is', 'i am here', 'this is', 
                           'i need to', 'i have to', 'i will', 'i went']
        if any(pattern in text.lower() for pattern in neutral_patterns) and confidence < 0.5:
            primary_emotion = "neutral"
            confidence = 0.6
        
        # Map GoEmotions labels to our response styles
        emotion_mapping = {
            'admiration': 'admiration',
            'amusement': 'joy',
            'anger': 'anger',
            'annoyance': 'anger',
            'approval': 'optimism',
            'caring': 'love',
            'confusion': 'confusion',
            'curiosity': 'curiosity',
            'desire': 'optimism',
            'disappointment': 'disappointment',
            'disapproval': 'anger',
            'disgust': 'disgust',
            'embarrassment': 'fear',
            'excitement': 'joy',
            'fear': 'fear',
            'gratitude': 'love',
            'grief': 'sadness',
            'joy': 'joy',
            'love': 'love',
            'nervousness': 'fear',
            'optimism': 'optimism',
            'pride': 'joy',
            'realization': 'surprise',
            'relief': 'joy',
            'remorse': 'sadness',
            'sadness': 'sadness',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        # Map to our response styles
        mapped_emotion = emotion_mapping.get(primary_emotion, primary_emotion)
        if mapped_emotion not in response_styles:
            mapped_emotion = 'other'
        
        return mapped_emotion, confidence, emotion_scores
        
    except Exception as e:
        logger.error(f"Error in emotion detection: {e}")
        return "neutral", 0.5, {}

def generate_reply(user_input: str, emotion: str, confidence: float) -> str:
    """Generate a contextual reply based on user input and detected emotion."""
    try:
        # Get the appropriate style instruction
        style_instruction = response_styles.get(emotion.lower(), response_styles["other"])
        
        # Adjust prompt based on confidence
        if confidence < 0.5:
            context_input = f"Respond naturally and warmly. User said: {user_input}"
        else:
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
            reply = f"I'm here to listen. Tell me more about what's on your mind."
            
        return reply

    except Exception as e:
        logger.error(f"Error generating reply: {e}")
        return "I'm here to listen and help. Can you tell me more?"

def process_audio_bytes(audio_bytes: bytes) -> str:
    """Process audio bytes and return transcription."""
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

def process_audio_file(uploaded_file) -> str:
    """Process uploaded audio file and return transcription."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        transcription = whisper_asr(tmp_path)
        os.unlink(tmp_path)
        return transcription.get("text", "").strip()
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return None

# ---------------------------------------------------- Main UI ---------------------------------
st.title("🧓 GoldenPal")
st.write("Providing companionship in your golden years, GoldenPal will respond with empathy and clear speech to your voice or text input.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot uses AI to:")
    st.write("• 🎭 Detect emotions with 28 categories")
    st.write("• 💬 Respond with empathy and understanding")
    st.write("• 🎤 Live Voice Input")
    st.write("• 📊 Better neutral & confusion detection")
    
    st.header("🔧 Status")
    if st.session_state.get('models_loaded', False):
        st.success("✅ All models loaded")
    else:
        st.error("❌ Models not loaded")
    
    st.header("🧪 Debug Mode")
    show_all_emotions = st.checkbox("Show all emotion scores", value=False)

# Input methods
st.subheader("🎤 Live Voice Input")
audio_bytes = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    format="wav",
    just_once=True,
    key="recorder"
)

st.subheader("📁 Upload Audio File")
uploaded_audio = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'flac', 'm4a']
)

st.subheader("⌨️ Text Input")
user_input = st.text_area(
    "Type your message here:",
    height=100,
    placeholder="Tell me how you're feeling or what's on your mind..."
)

# Submit button
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
            detected_emotion, confidence, all_emotions = detect_emotion_advanced(final_input)
            reply = generate_reply(final_input, detected_emotion, confidence)

        # Show emotion + reply
        col1, col2 = st.columns(2)
        col1.metric("Detected Emotion", detected_emotion.title())
        col2.metric("Confidence", f"{confidence:.1%}")

        # Show all emotions in debug mode
        if show_all_emotions and all_emotions:
            st.subheader("📊 All Detected Emotions")
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:5]
            for emotion, score in sorted_emotions:
                st.write(f"**{emotion}**: {score:.2%}")

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