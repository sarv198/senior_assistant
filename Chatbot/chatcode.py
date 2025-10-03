import streamlit as st
import os
import tempfile
import logging
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_mic_recorder import mic_recorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotion response styles
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
    "curiosity": "Engage with interest and provide helpful information.",
    "other": "Offer a gentle, open-ended response."
}

# Emotion mapping from GoEmotions to response styles
emotion_map = {
    'admiration': 'joy', 'amusement': 'joy', 'anger': 'anger', 'annoyance': 'anger',
    'approval': 'optimism', 'caring': 'love', 'confusion': 'confusion', 'curiosity': 'curiosity',
    'desire': 'optimism', 'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'anger',
    'embarrassment': 'fear', 'excitement': 'joy', 'fear': 'fear', 'gratitude': 'love',
    'grief': 'sadness', 'joy': 'joy', 'love': 'love', 'nervousness': 'fear',
    'optimism': 'optimism', 'pride': 'joy', 'realization': 'surprise', 'relief': 'joy',
    'remorse': 'sadness', 'sadness': 'sadness', 'surprise': 'surprise', 'neutral': 'neutral'
}

st.set_page_config(page_title="Senior Companion Chatbot", layout="centered", initial_sidebar_state="collapsed")

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load all AI models."""
    try:
        emotion_clf = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=-1, top_k=None)
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        chat_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill", torch_dtype=torch.float32, low_cpu_mem_usage=True)
        whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1, chunk_length_s=30)
        return emotion_clf, tokenizer, chat_model, whisper
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None, None, None, None

# Initialize models
if 'models_loaded' not in st.session_state:
    with st.spinner("Loading AI models..."):
        st.session_state.emotion_clf, st.session_state.tokenizer, st.session_state.chat_model, st.session_state.whisper = load_models()
        st.session_state.models_loaded = True

emotion_clf = st.session_state.emotion_clf
tokenizer = st.session_state.tokenizer
chat_model = st.session_state.chat_model
whisper = st.session_state.whisper

def detect_emotion(text):
    """Detect emotion with improved neutral/confusion handling."""
    try:
        results = emotion_clf(text)
        
        # Handle both list and dict formats from the classifier
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                scores = {item['label']: item['score'] for item in results[0]}
            else:
                scores = {item['label']: item['score'] for item in results}
        else:
            scores = {}
        
        if not scores:
            return "neutral", None, 0.5, {}
        
        # Get top 2 emotions
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        emotion1, conf1 = sorted_emotions[0]
        emotion2, conf2 = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0)
        
        # Handle low confidence
        if conf1 < 0.4:
            question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', '?']
            emotion1 = "curiosity" if any(w in text.lower() for w in question_words) else "neutral"
            emotion2 = None
        
        # Detect confusion
        confusion_words = ['confused', 'don\'t understand', 'not sure', 'unclear', 'what do you mean', 'huh']
        if any(w in text.lower() for w in confusion_words):
            emotion1, conf1 = "confusion", max(conf1, 0.7)
        
        # Detect neutral statements (only very obvious factual statements)
        neutral_patterns = ['the weather is', 'today is', 'the time is', 'it is located']
        if any(p in text.lower() for p in neutral_patterns) and conf1 < 0.4:
            emotion1, conf1 = "neutral", 0.6
            emotion2 = None
        
        # Map to response styles
        mapped1 = emotion_map.get(emotion1, 'other')
        mapped1 = mapped1 if mapped1 in response_styles else 'other'
        
        mapped2 = None
        if emotion2 and conf2 > 0.2:
            mapped2 = emotion_map.get(emotion2, 'other')
            mapped2 = mapped2 if mapped2 in response_styles else None
            if mapped2 == mapped1:
                mapped2 = None
        
        return mapped1, mapped2, conf1, scores
        
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        return "neutral", None, 0.5, {}
        
def generate_reply(text, emotion1, emotion2, conf):
    """Generate empathetic reply."""
    try:
        style1 = response_styles.get(emotion1.lower(), response_styles["other"])
        
        if emotion2:
            style2 = response_styles.get(emotion2.lower(), response_styles["other"])
            context = f"The user feels {emotion1} and {emotion2}. {style1} Also, {style2} User said: {text}"
        elif conf < 0.5:
            context = f"Respond naturally and warmly. User said: {text}"
        else:
            context = f"The user feels {emotion1}. {style1} User said: {text}"
        
        inputs = tokenizer([context], return_tensors="pt", truncation=True, max_length=512)
        reply_ids = chat_model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        
        if "User said:" in reply:
            reply = reply.split("User said:")[-1].strip()
        
        return reply if len(reply.strip()) >= 3 else "I'm here to listen. Tell me more about what's on your mind."
    except Exception as e:
        logger.error(f"Reply generation error: {e}")
        return "I'm here to listen and help. Can you tell me more?"

def process_audio(audio_data, is_bytes=True):
    """Process audio and return transcription."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data if is_bytes else audio_data.read())
            tmp_path = tmp.name
        result = whisper(tmp_path)
        os.unlink(tmp_path)
        return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return None

# UI
st.title("🧓 GoldenPal")
st.write("Providing companionship in your golden years with empathetic AI responses.")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("• 🎭 Detects 28 emotion categories\n• 💬 Empathetic responses\n• 🎤 Voice input support\n• 📊 Better neutral/confusion detection")
    st.header("🔧 Status")
    st.success("✅ All models loaded" if st.session_state.get('models_loaded') else "❌ Models not loaded")
    show_debug = st.checkbox("Show all emotion scores", value=False)

st.subheader("🎤 Live Voice Input")
audio_bytes = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", format="wav", just_once=True, key="recorder")

st.subheader("📁 Upload Audio File")
uploaded_audio = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac', 'm4a'])

st.subheader("⌨️ Text Input")
user_input = st.text_area("Type your message here:", height=100, placeholder="Tell me how you're feeling or what's on your mind...")

if st.button("🚀 Submit", type="primary", use_container_width=True):
    final_input = None

    if audio_bytes:
        st.info("🎤 Processing mic recording...")
        final_input = process_audio(audio_bytes, is_bytes=True)
        if final_input:
            st.write(f"**Transcribed (mic):** {final_input}")
        else:
            st.error("❌ Failed to transcribe mic recording.")

    elif uploaded_audio:
        st.info("📁 Processing uploaded file...")
        final_input = process_audio(uploaded_audio, is_bytes=False)
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
            emotion1, emotion2, conf, all_scores = detect_emotion(final_input)
            reply = generate_reply(final_input, emotion1, emotion2, conf)

        col1, col2 = st.columns(2)
        emotion_display = f"{emotion1.title()}" + (f" + {emotion2.title()}" if emotion2 else "")
        col1.metric("Detected Emotion(s)", emotion_display)
        col2.metric("Confidence", f"{conf:.1%}")

        if show_debug and all_scores:
            st.subheader("📊 Top 5 Emotions")
            for emo, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"**{emo}**: {score:.2%}")

        st.markdown(f"""
            <div style='font-size:26px; color:#2E8B57; padding:20px; 
            background-color:#f0f8f0; border-radius:10px; 
            border-left: 5px solid #2E8B57;'>
            <strong>{reply}</strong>
            </div>
            """, unsafe_allow_html=True)