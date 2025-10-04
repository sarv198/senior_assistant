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

# Emotion emojis
emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😰",
    "surprise": "😲",
    "love": "❤️",
    "confusion": "😕",
    "optimism": "🌟",
    "neutral": "😐",
    "curiosity": "🤔",
    "other": "💭"
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
        
        # Check for mixed emotions - if top emotions are close in score, it's complex
        if emotion2 and conf2 > 0.25 and (conf1 - conf2) < 0.15:
            # Emotions are closely scored - this is a complex emotional state
            pass  # Keep both emotions
        
        # Detect life confusion/overwhelm patterns
        overwhelm_words = ['all over the place', 'overwhelmed', 'don\'t know what', 'no clear path', 
                          'confused about', 'lost', 'stuck', 'no direction']
        if any(w in text.lower() for w in overwhelm_words):
            # Override with confusion if not already top emotion
            if emotion1 not in ['confusion', 'nervousness', 'disappointment']:
                emotion1 = "confusion"
                conf1 = max(conf1, 0.75)
        
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
        
        # Build a more conversational prompt that discourages echoing
        if emotion2:
            style2 = response_styles.get(emotion2.lower(), response_styles["other"])
            context = (
                f"You are a warm, caring companion having a conversation. "
                f"The person seems to feel {emotion1} and {emotion2}. {style1} {style2} "
                f"Respond directly to them in a natural, conversational way without repeating what they said. "
                f"Their message: {text}\nYour response:"
            )
        elif conf < 0.5:
            context = (
                f"You are a warm, caring companion. "
                f"Respond naturally and warmly in your own words. "
                f"Their message: {text}\nYour response:"
            )
        else:
            context = (
                f"You are a warm, caring companion having a conversation. "
                f"The person seems to feel {emotion1}. {style1} "
                f"Respond directly to them in a natural way without repeating what they said. "
                f"Their message: {text}\nYour response:"
            )
        
        # Tokenize with appropriate settings
        inputs = tokenizer([context], return_tensors="pt", truncation=True, max_length=512)
        
        # Generate with adjusted parameters for more natural responses
        reply_ids = chat_model.generate(
            **inputs, 
            max_length=120,  # Slightly longer for complete thoughts
            min_length=15,   # Ensure substantial responses
            do_sample=True, 
            temperature=0.8,  # Slightly higher for more natural variation
            top_p=0.9,       # Nucleus sampling for better quality
            repetition_penalty=1.3,  # Discourage repetition
            no_repeat_ngram_size=3,  # Prevent repeating 3-word phrases
            pad_token_id=tokenizer.eos_token_id
        )
        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        
        # More aggressive cleaning to remove echoing
        # Remove common echo patterns
        echo_patterns = [
            "User said:", "Their message:", "Your response:", 
            "The person", "They said", "You said"
        ]
        for pattern in echo_patterns:
            if pattern in reply:
                reply = reply.split(pattern)[-1].strip()
        
        # Remove if reply starts by repeating user's text
        user_words = text.lower().split()[:5]  # First 5 words of user input
        reply_words = reply.lower().split()[:5]
        
        # If the reply starts with similar words to input, it's likely echoing
        overlap = sum(1 for word in reply_words if word in user_words)
        if overlap >= 3 and len(reply_words) >= 3:
            # Try to extract the actual response after the echo
            sentences = reply.split('.')
            if len(sentences) > 1:
                reply = '.'.join(sentences[1:]).strip()
        
        # Remove quotes if the model quoted the user
        reply = reply.replace('"', '').replace("'", "").strip()
        
        # Ensure reply doesn't start with repetitive phrases
        repetitive_starts = [
            text.lower()[:20],  # First 20 chars of user input
            "i feel", "i am feeling", "you feel", "you are feeling"
        ]
        for start in repetitive_starts:
            if reply.lower().startswith(start):
                # Extract everything after the first sentence
                sentences = reply.split('.')
                if len(sentences) > 1:
                    reply = '.'.join(sentences[1:]).strip()
                break
        
        # Ensure quality and length
        if len(reply.strip()) < 10:
            return "I'm here for you. What's on your mind right now?"
        
        # Make sure reply ends properly
        if reply and not reply[-1] in '.!?':
            reply += '.'
            
        return reply
        
    except Exception as e:
        logger.error(f"Reply generation error: {e}")
        return "I'm here to listen and help. Can you tell me more?"

def process_audio(audio_data, is_bytes=True):
    """Process audio and return transcription."""
    try:
        if audio_data is None:
            logger.error("Audio data is None")
            return None
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data if is_bytes else audio_data.read())
            tmp_path = tmp.name
        
        logger.info(f"Processing audio file: {tmp_path}")
        
        if whisper is None:
            logger.error("Whisper model not loaded")
            os.unlink(tmp_path)
            return None
            
        result = whisper(tmp_path)
        logger.info(f"Transcription result: {result}")
        os.unlink(tmp_path)
        
        text = result.get("text", "").strip()
        logger.info(f"Extracted text: {text}")
        return text if text else None
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}", exc_info=True)
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
audio_bytes = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", format="wav", just_once=False, key="recorder")

if audio_bytes:
    st.success(f"✅ Audio recorded: {len(audio_bytes['bytes'])} bytes")
    if st.button("🎯 Transcribe Recording", type="secondary"):
        with st.spinner("🎤 Transcribing your voice..."):
            final_input = process_audio(audio_bytes['bytes'], is_bytes=True)
            if final_input:
                st.session_state.transcribed_text = final_input
                st.success(f"**Transcribed:** {final_input}")
            else:
                st.error("❌ Failed to transcribe. Try again or check terminal for errors.")

st.subheader("📁 Upload Audio File")
uploaded_audio = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac', 'm4a'])

st.subheader("⌨️ Text Input")
user_input = st.text_area(
    "Type your message here:", 
    height=100, 
    placeholder="Tell me how you're feeling or what's on your mind...",
    value=st.session_state.get('transcribed_text', '')
)

if st.button("🚀 Submit", type="primary", use_container_width=True):
    final_input = None

    if uploaded_audio:
        st.info("📁 Processing uploaded file...")
        final_input = process_audio(uploaded_audio, is_bytes=False)
        if final_input:
            st.write(f"**Transcribed (file):** {final_input}")
        else:
            st.error("❌ Failed to transcribe uploaded audio.")

    elif user_input and user_input.strip():
        final_input = user_input.strip()
    else:
        st.warning("⚠️ Please provide input (record voice, upload audio file, or type text).")

    if final_input:
        with st.spinner("🤖 Analyzing your message..."):
            emotion1, emotion2, conf, all_scores = detect_emotion(final_input)
            reply = generate_reply(final_input, emotion1, emotion2, conf)

        col1, col2 = st.columns(2)
        emoji1 = emotion_emojis.get(emotion1, "💭")
        emoji2 = emotion_emojis.get(emotion2, "") if emotion2 else ""
        emotion_display = f"{emoji1} {emotion1.title()}" + (f" + {emoji2} {emotion2.title()}" if emotion2 else "")
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