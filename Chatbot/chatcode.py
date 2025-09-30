# Imports
# streamlit for UI, os for temporary files
import streamlit as st
import os
import tempfile
import logging
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

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

# Check if imports were successful, if not, app is stopped
if not IMPORTS_SUCCESS:
    st.stop()

# Model loading functions with better error handling and optimization
@st.cache_resource(show_spinner="Loading your personal companion...")
def load_emotion_model():
    """Load emotion classification model with fallback options."""
    try:
        return pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            device=-1,  # Force CPU usage
            return_all_scores=False
        )
    except Exception as e:
        logger.error(f"Failed to load emotion model: {e}")
        return None

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
        emotion_classifier = load_emotion_model()
        chat_tokenizer, chat_model = load_chatbot()
        whisper_asr = load_whisper()
        
        # Check if all models loaded successfully
        if emotion_classifier and chat_tokenizer and chat_model and whisper_asr:
            st.session_state.models_loaded = True
            st.session_state.emotion_classifier = emotion_classifier
            st.session_state.chat_tokenizer = chat_tokenizer
            st.session_state.chat_model = chat_model
            st.session_state.whisper_asr = whisper_asr
            st.success("✅ All models loaded successfully!")
        else:
            st.error("❌ Failed to load some models. Please try again, sorry!")
            st.stop()
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

def process_audio_file(audio_data, is_uploaded: bool = False) -> str:
    """Process audio file and return transcription."""
    try:
        if is_uploaded and audio_data is not None:
            # For uploaded files, uploaded bytes saved to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data.read())
                tmp_file_path = tmp.name
            transcription = whisper_asr(audio_data.read())
            os.unlink(tmp_path)
        elif not is_uploaded and audio_data is not None:
            # For recorded audio, save to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                transcription = whisper_asr(tmp_file_path)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        else:
            return None
            
        return transcription.get("text", "").strip()
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

# Main UI
st.title("🧓 Senior Companion Chatbot")
st.write("Talk or type to the bot. It will respond with empathy and clear speech.")

# Add a sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot uses AI to:")
    st.write("• 🎭 Detect emotions in your messages")
    st.write("• 💬 Respond with empathy and understanding")
    st.write("• 🎤 Convert speech to text")
    st.write("• 🔊 Speak responses back to you")
    
    st.header("🔧 Status")
    if st.session_state.get('models_loaded', False):
        st.success("✅ All models loaded")
    else:
        st.error("❌ Models not loaded")

# Input methods
st.subheader("🎤 Voice Input")
st.write("Record your message or upload an audio file:")

# Create columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    try:
        # Record button with error handling
        audio_bytes = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            key='recorder'
        )
    except Exception as e:
        st.error(f"Microphone recording not available: {e}")
        audio_bytes = None

with col2:
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.success("✅ Recording captured! Click Submit to process.")

# Alternative: Upload audio file
st.write("---")
st.write("**📁 Or upload an audio file:**")
uploaded_audio = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'flac', 'm4a'], 
    key='upload',
    help="Supported formats: WAV, MP3, FLAC, M4A"
)

# Text input
st.subheader("⌨️ Text Input")
user_input = st.text_area(
    "Type your message here:", 
    height=100,
    placeholder="Tell me how you're feeling or what's on your mind..."
)

# Process input button
if st.button("🚀 Submit", type="primary", use_container_width=True):
    final_input = ""
    
    # Process different input types
    if audio_bytes:
        st.info("🎤 Processing recorded audio...")
        final_input = process_audio_file(audio_bytes, is_uploaded=False)
        if final_input:
            st.write(f"**Transcribed:** {final_input}")
        else:
            st.error("❌ Failed to transcribe audio. Please try again.")
            
    elif uploaded_audio:
        st.info("📁 Processing uploaded audio...")
        final_input = process_audio_file(uploaded_audio, is_uploaded=True)
        if final_input:
            st.write(f"**Transcribed:** {final_input}")
        else:
            st.error("❌ Failed to transcribe audio. Please try again.")
            
    elif user_input and user_input.strip():
        final_input = user_input.strip()
    else:
        st.warning("⚠️ Please provide some input (voice, audio file, or text).")

    # Process the input if we have it
    if final_input:
        try:
            with st.spinner("🤖 Analyzing your message..."):
                # Step 1: Detect emotion
                emotion_result = emotion_classifier(final_input, top_k=1)[0]
                detected_emotion = emotion_result['label']
                confidence = emotion_result['score']

                # Step 2: Generate reply using emotion-aware context
                reply = generate_reply(final_input, detected_emotion)

            # Display results
            st.success("✅ Analysis complete!")
            
            # Show detected emotion and confidence
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Emotion", detected_emotion.title())
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")

            # Show reply in large, readable font
            st.markdown("### 🤖 Bot Response:")
            st.markdown(f"<div style='font-size:24px; color: #2E8B57; padding: 20px; background-color: #f0f8f0; border-radius: 10px; border-left: 5px solid #2E8B57;'><strong>{reply}</strong></div>", 
                       unsafe_allow_html=True)

            # Generate and play audio
            with st.spinner("🔊 Generating speech..."):
                audio_file_path = speak_text(reply)
                
            if audio_file_path:
                st.audio(audio_file_path, format="audio/mp3")
                # Clean up audio file
                try:
                    if os.path.exists(audio_file_path):
                        os.unlink(audio_file_path)
                except Exception as e:
                    logger.warning(f"Could not clean up audio file: {e}")
            else:
                st.warning("⚠️ Could not generate audio. Text response is still available.")
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            st.error(f"❌ An error occurred while processing your message: {str(e)}")
            st.write("Please try again or check your input.")

# Add footer
st.write("---")
st.write("💡 **Tip:** The bot works best with clear speech and complete sentences.")