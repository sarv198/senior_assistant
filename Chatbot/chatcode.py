import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import os, tempfile
from streamlit_mic_recorder import mic_recorder

# Core chatbot code
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

#2. Load Hugging Face Models
st.set_page_config(page_title = "Senior Companion Chatbot", layout = "centered")

# a) We load a DistilBERT model fine-tuned on Google's GoEmotions dataset
#   - classifies text into one of 10 emotion categories and creates according tag
@st.cache_resource
def load_emotion_model():
  return pipeline("text-classification",
                  model = "bhadresh-savani/distilbert-base-uncased-emotion")

# b) BlenderBot 400M Distill: lightweight, conversational model
#  - converts user text into numbers to predict reply (in readable text)
@st.cache_resource
def load_chatbot():
  model_name = "facebook/blenderbot-400M-distill"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  return tokenizer, model

# c) Load Whisper for speech to text transcription
@st.cache_resource
def load_whisper():
  return pipeline("automatic-speech-recognition", model = "openai/whisper-small")

emotion_classifier = load_emotion_model()
chat_tokenizer, chat_model = load_chatbot()
whisper_asr = load_whisper()

#3. prepend the emotion tag to input
def generate_reply(user_input, emotion):
    # Get the appropoiate style instruction based on detected emotion
    style_instruction = response_styles.get(emotion.lower(), response_styles["other"])
    context_input = f"The user feels {emotion}. {style_instruction} User said: {user_input}"
    inputs = chat_tokenizer([context_input], return_tensors="pt")
    reply_ids = chat_model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
    reply = chat_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    if "User said:" in reply:
      reply = reply.split("User said:")[-1].strip()

    return reply

#4. Text-to-Speech (TTS)
def speak_text(text):
  tts = gTTS(text=text, lang="en")
  tts.save("reply.mp3")
  return "reply.mp3"

#5 Stremlit UI (large-font, playable audio of bot's voice)
st.title("🧓 Senior Companion Chatbot")
st.write("Talk or type to the bot. It will respond with empathy and clear speech.")


#6. a) Voice input using Whisper small model
st.subheader("🎤 Speak to the bot")

# c) Creating columns for better layout
col1, col2 = st.columns([1,3])

with col1:
  # Record button
  audio_bytes = mic_recorder(
      start_prompt = "🎤 Start Recording",
      stop_prompt = "⏹️ Stop Recording",
      just_once=False,
      use_container_width=True,
      format="wav",
      key='recorder'
  )

with col2:
  if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.success("✅ Recording captured! Click Submit to process.")

# Alternative method if streamlit-mic-recorder is not available
st.write("---")
st.write("**Alternative: Upload audio file**")
uploaded_audio = st.file_uploader("Or upload an audio file", type=['wav', 'mp3', 'flac'], key='upload')

st.subheader("⌨️ Or type your message")
user_input = st.text_input("Type here:")

#7. Process input
if st.button("Submit"):
  final_input = ""
  tmp_file_path = None

  # check for recorded audio
  if audio_bytes:
    # save audio bytes to temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
      tmp_file.write(audio_bytes)
      tmp_file_path = tmp_file.name
    
    try:
      # Step 1: transcribe recorded audio with Whisper
      transcription = whisper_asr(tmp_file_path)
      final_input = transcription["text"]
      st.write(f"**Transcribed:** {final_input}")
    finally:
      # Clean up temporary file
      if tmp_file_path and os.path.exists(tmp_file_path):
        os.unlink(tmp_file_path)

  elif uploaded_audio:
    # step 1: transcribe uploaded audio with Whisper
    transcription = whisper_asr(uploaded_audio)
    final_input = transcription["text"]
    st.write(f"**Transcribed:** {final_input}")
  elif user_input:
    final_input = user_input

if final_input:
  try:
    # step 2: Detect emotion
    emotion_result = emotion_classifier(final_input, top_k=1)[0]
    detected_emotion = emotion_result['label']
    confidence = emotion_result['score']

    # Step 3: Generate reply using emotion-aware context
    reply = generate_reply(final_input, detected_emotion)

    # Show detected emotion and confidence
    st.write(f"**Detected emotion:** {detected_emotion} (confidence: {confidence:.2f})")

    # Step 4: Show reply in large font
    st.markdown(f"<p style='font-size:28px; color: #2E8B57;'><strong>Bot:</strong> {reply}</p>",
                     unsafe_allow_html=True)

    # Step 5: Play audio
    audio_file_path = speak_text(reply)
    st.audio(audio_file_path)
    
    # Clean up audio file
    if os.path.exists(audio_file_path):
      os.unlink(audio_file_path)
      
  except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please try again or check your input.")