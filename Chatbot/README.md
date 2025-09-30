# Senior Companion Chatbot

An AI-powered chatbot designed specifically for seniors, featuring emotion detection, voice interaction, and empathetic responses.

## Features

- 🎭 **Emotion Detection**: Automatically detects emotions in user messages
- 💬 **Empathetic Responses**: Generates contextually appropriate responses based on detected emotions
- 🎤 **Voice Input**: Record audio messages or upload audio files
- 🔊 **Text-to-Speech**: Converts bot responses to audio
- 🎨 **Senior-Friendly UI**: Large fonts, clear layout, and intuitive design

## Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - DistilBERT for emotion classification
  - BlenderBot for conversational responses
  - Whisper for speech-to-text
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Audio Processing**: librosa, soundfile

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd senior_assistant/Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python test_app.py
```

4. Run the application:
```bash
streamlit run chatcode.py
```

### Streamlit Cloud Deployment

1. **Push to GitHub**: Ensure your code is in a GitHub repository

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository

3. **Configure Deployment**:
   - **Main file path**: `Chatbot/chatcode.py`
   - **Python version**: 3.9 or higher
   - **Dependencies**: The app will automatically install from `requirements.txt`

4. **Deploy**: Click "Deploy" and wait for the build to complete

## Troubleshooting

### Common Issues

1. **503 Service Unavailable Errors**:
   - These usually indicate model loading timeouts
   - The app now includes better error handling and progress indicators
   - Models are cached after first load for faster subsequent runs

2. **Audio Processing Issues**:
   - Ensure microphone permissions are granted
   - Try uploading audio files instead of recording
   - Check that audio files are in supported formats (WAV, MP3, FLAC, M4A)

3. **Model Loading Failures**:
   - The app includes fallback mechanisms
   - Check the sidebar status indicator
   - Refresh the page if models fail to load

### Performance Optimization

- **First Load**: Initial model loading may take 2-5 minutes
- **Subsequent Loads**: Models are cached for faster performance
- **Memory Usage**: Optimized for Streamlit Cloud's memory limits
- **CPU Usage**: All models run on CPU for compatibility

## File Structure

```
Chatbot/
├── chatcode.py              # Main application
├── requirements.txt         # Python dependencies
├── test_app.py             # Local testing script
├── .streamlit/
│   └── config.toml         # Streamlit configuration
└── README.md               # This file
```

## Usage

1. **Voice Input**: Click the microphone button to record, or upload an audio file
2. **Text Input**: Type your message in the text area
3. **Submit**: Click the "Submit" button to process your input
4. **Response**: The bot will:
   - Detect your emotion
   - Generate an empathetic response
   - Provide audio playback of the response

## Emotion Categories

The bot recognizes these emotions and responds accordingly:
- Joy, Sadness, Anger, Fear, Surprise
- Love, Confusion, Optimism, Neutral, Other

## Support

If you encounter issues:
1. Check the sidebar status indicators
2. Try refreshing the page
3. Ensure you have a stable internet connection
4. For deployment issues, check the Streamlit Cloud logs

## License

This project is open source and available under the MIT License.
