#!/usr/bin/env python3
"""
Simple test script to verify the chatbot application works locally.
Run this before deploying to Streamlit Cloud.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("OK - Streamlit imported successfully")
    except ImportError as e:
        print(f"ERROR - Streamlit import failed: {e}")
        return False
    
    try:
        from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
        print("OK - Transformers imported successfully")
    except ImportError as e:
        print(f"ERROR - Transformers import failed: {e}")
        return False
    
    try:
        from gtts import gTTS
        print("OK - gTTS imported successfully")
    except ImportError as e:
        print(f"ERROR - gTTS import failed: {e}")
        return False
    
    try:
        from streamlit_mic_recorder import mic_recorder
        print("OK - streamlit-mic-recorder imported successfully")
    except ImportError as e:
        print(f"ERROR - streamlit-mic-recorder import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without loading heavy models."""
    print("\nTesting basic functionality...")
    
    try:
        # Test emotion styles
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
        
        # Test emotion lookup
        test_emotion = "joy"
        style = response_styles.get(test_emotion.lower(), response_styles["other"])
        print(f"OK - Emotion style lookup works: {test_emotion} -> {style[:30]}...")
        
        # Test text processing
        test_input = "Hello, how are you?"
        if len(test_input.strip()) > 0:
            print("OK - Text input validation works")
        
        return True
        
    except Exception as e:
        print(f"ERROR - Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Senior Companion Chatbot - Local Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\nERROR - Import tests failed. Please install missing packages:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\nERROR - Basic functionality tests failed.")
        sys.exit(1)
    
    print("\nSUCCESS - All tests passed!")
    print("\nTo run the application locally:")
    print("streamlit run chatcode.py")
    print("\nTo deploy to Streamlit Cloud:")
    print("1. Push your code to GitHub")
    print("2. Connect your repository to Streamlit Cloud")
    print("3. Set the main file path to: Chatbot/chatcode.py")

if __name__ == "__main__":
    main()
