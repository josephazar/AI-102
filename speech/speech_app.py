"""
Azure Speech Services - Streamlit Application
--------------------------------------
This module provides a Streamlit interface for demonstrating Azure Speech Services capabilities.
"""

import os
import time
import tempfile
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import base64
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Speech Services
SPEECH_KEY = os.getenv('SPEECH_KEY')
SPEECH_REGION = os.getenv('SPEECH_REGION')

# Function to create a download link for audio files
def get_audio_download_link(audio_bytes, filename, text):
    """Generates a link to download the audio file"""
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to check if Speech Service is configured
def is_speech_configured():
    """Check if Speech Service credentials are properly configured"""
    if not SPEECH_KEY or not SPEECH_REGION:
        st.error("⚠️ Speech Services credentials not found. Please add SPEECH_KEY and SPEECH_REGION to your .env file.")
        st.info("Get your Speech key and region from the Azure portal.")
        return False
    return True

# Speech-to-Text (Recognition) functions
def recognize_from_microphone():
    """Recognize speech from microphone and return the transcription result"""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    
    # Get selected language
    selected_lang = st.session_state.get('recognition_language', 'en-US')
    speech_config.speech_recognition_language = selected_lang
    
    # Use default microphone as audio input
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Set up result container
    result_placeholder = st.empty()
    result_placeholder.info("Listening... Speak now.")
    
    # Use a simpler approach with recognize_once() instead of continuous recognition
    # This is more reliable for short phrases and simple use cases
    result = speech_recognizer.recognize_once()
    
    # Process the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        result_placeholder.success(f"Recognized: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        result_placeholder.warning("No speech detected or couldn't recognize speech.")
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        result_placeholder.error(f"Recognition canceled: {result.cancellation_details.reason}")
        return ""
    
    return ""

def recognize_from_file(audio_file):
    """Recognize speech from uploaded audio file"""
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getbuffer())
        tmp_path = tmp_file.name
    
    # Configure speech recognition
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    
    # Get selected language
    selected_lang = st.session_state.get('recognition_language', 'en-US')
    speech_config.speech_recognition_language = selected_lang
    
    # Use the temp file as audio input
    audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
    
    # Create speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Start speech recognition
    result = speech_recognizer.recognize_once()
    
    # Delete the temp file
    try:
        os.unlink(tmp_path)
    except:
        pass
    
    # Process and return result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        st.error(f"No speech could be recognized: {result.no_match_details}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        st.error(f"Recognition canceled: {result.cancellation_details.reason}")
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            st.error(f"Error details: {result.cancellation_details.error_details}")
    
    return None

# Text-to-Speech (Synthesis) functions
def synthesize_speech(text, voice_name):
    """Synthesize speech from text and return audio bytes"""
    if not text:
        return None
    
    # Configure speech synthesis
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = voice_name
    
    # Create a temp file for audio output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_path = tmp_file.name
    
    # Configure audio output to file
    audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
    
    # Create speech synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    # Synthesize speech
    result = speech_synthesizer.speak_text_async(text).get()
    
    # Check result
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        st.error(f"Speech synthesis failed: {result.cancellation_details.reason}")
        try:
            os.unlink(tmp_path)
        except:
            pass
        return None
    
    # Read the audio file into bytes
    with open(tmp_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    
    # Clean up
    try:
        os.unlink(tmp_path)
    except:
        pass
    
    return audio_bytes

# Speech Translation functions
def translate_speech(target_languages):
    """Translate speech from microphone to target languages"""
    if not target_languages:
        st.error("Please select at least one target language.")
        return None, None
    
    # Configure speech translation
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=SPEECH_KEY, region=SPEECH_REGION
    )
    
    # Set source language
    source_language = st.session_state.get('source_language', 'en-US')
    translation_config.speech_recognition_language = source_language
    
    # Add target languages
    for lang in target_languages:
        translation_config.add_target_language(lang)
    
    # Use default microphone
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create translation recognizer
    translator = speechsdk.translation.TranslationRecognizer(translation_config=translation_config, audio_config=audio_config)
    
    # Set up result placeholder
    result_placeholder = st.empty()
    result_placeholder.info("Listening for speech to translate... Speak now.")
    
    # Use recognize_once for simpler translation
    result = translator.recognize_once()
    
    # Process result
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        result_placeholder.success(f"Recognized: {result.text}")
        return result.text, result.translations
    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        result_placeholder.warning("Speech recognized, but couldn't be translated.")
        return result.text, None
    elif result.reason == speechsdk.ResultReason.NoMatch:
        result_placeholder.warning("No speech detected or couldn't recognize speech.")
        return None, None
    elif result.reason == speechsdk.ResultReason.Canceled:
        result_placeholder.error(f"Translation canceled: {result.cancellation_details.reason}")
        return None, None
    
    return None, None

# Voices data for different languages
def get_voice_options():
    """Return voice options organized by language"""
    return {
        "English": [
            "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", 
            "en-GB-SoniaNeural", "en-GB-RyanNeural"
        ],
        "Spanish": [
            "es-ES-ElviraNeural", "es-ES-AlvaroNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural"
        ],
        "French": [
            "fr-FR-DeniseNeural", "fr-FR-HenriNeural", "fr-CA-SylvieNeural", "fr-CA-JeanNeural"
        ],
        "German": [
            "de-DE-KatjaNeural", "de-DE-ConradNeural"
        ],
        "Italian": [
            "it-IT-ElsaNeural", "it-IT-DiegoNeural"
        ],
        "Portuguese": [
            "pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", "pt-PT-RaquelNeural", "pt-PT-DuarteNeural"
        ],
        "Japanese": [
            "ja-JP-NanamiNeural", "ja-JP-KeitaNeural"
        ],
        "Chinese": [
            "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-TW-HsiaoChenNeural", "zh-TW-YunJheNeural"
        ],
        "Arabic": [
            "ar-SA-ZariyahNeural", "ar-SA-HamedNeural"
        ],
        "Russian": [
            "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"
        ]
    }

# Language options for speech recognition and translation
def get_recognition_languages():
    """Return language options for speech recognition"""
    return [
        "en-US", "en-GB", "es-ES", "es-MX", "fr-FR", "fr-CA", 
        "de-DE", "it-IT", "pt-BR", "pt-PT", "ja-JP", 
        "zh-CN", "zh-TW", "ar-SA", "ru-RU"
    ]

def get_language_name(code):
    """Convert language code to readable name"""
    language_names = {
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "es-ES": "Spanish (Spain)",
        "es-MX": "Spanish (Mexico)",
        "fr-FR": "French (France)",
        "fr-CA": "French (Canada)",
        "de-DE": "German",
        "it-IT": "Italian",
        "pt-BR": "Portuguese (Brazil)",
        "pt-PT": "Portuguese (Portugal)",
        "ja-JP": "Japanese",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ar-SA": "Arabic",
        "ru-RU": "Russian"
    }
    return language_names.get(code, code)

# Main application
def show_speech_services():
    """Main function to show Speech Services demo"""
    st.title("Azure Speech Services")
    st.markdown("""
    Explore the capabilities of Azure Speech Services - Microsoft's cloud-based API for speech recognition, 
    synthesis, and translation. This demo showcases key features including speech-to-text, text-to-speech, 
    and real-time speech translation.
    """)
    
    # Check if Speech Service is configured
    if not is_speech_configured():
        # Show instructions for adding environment variables
        st.markdown("""
        ### Setup Instructions
        
        To use this demo, you need to add the following environment variables to your `.env` file:
        
        ```
        SPEECH_KEY=your_speech_service_key
        SPEECH_REGION=your_speech_service_region
        ```
        
        You can get these values from the Azure portal by:
        1. Creating a Speech resource in the Azure portal
        2. Going to "Keys and Endpoint" in your Speech resource
        3. Copying the Key and Region values
        """)
        return
    
    # Create tabs for each functionality
    tab1, tab2, tab3, tab4 = st.tabs([
        "Speech-to-Text", "Text-to-Speech", "Speech Translation", "About Speech Services"
    ])
    
    # Tab 1: Speech-to-Text
    with tab1:
        st.header("Speech Recognition")
        st.markdown("""
        Convert spoken audio to text using Azure's state-of-the-art speech recognition technology. 
        You can either speak into your microphone or upload an audio file.
        """)
        
        # Language selection
        recognition_languages = get_recognition_languages()
        selected_language = st.selectbox(
            "Select recognition language",
            recognition_languages,
            format_func=get_language_name,
            key="recognition_language",
            index=0
        )
        
        # Input method selection
        input_method = st.radio("Choose input method", ["Microphone", "Upload Audio File"])
        
        transcription_result = None
        
        if input_method == "Microphone":
            if st.button("Start Recording", key="start_recording"):
                with st.spinner("Listening..."):
                    transcription_result = recognize_from_microphone()
        else:
            uploaded_file = st.file_uploader("Upload audio file (WAV format recommended)", type=["wav", "mp3", "ogg"])
            if uploaded_file is not None:
                with st.spinner("Transcribing..."):
                    transcription_result = recognize_from_file(uploaded_file)
        
        # Display transcription result
        if transcription_result:
            st.subheader("Transcription Result")
            st.success(transcription_result)
            
            # Copy to clipboard button
            st.markdown(f"""
            <textarea id="transcription-result" style="position: absolute; left: -9999px;">{transcription_result}</textarea>
            <button onclick="navigator.clipboard.writeText(document.getElementById('transcription-result').value); alert('Copied to clipboard!');">
                Copy to Clipboard
            </button>
            """, unsafe_allow_html=True)
            
            # Option to use in text-to-speech
            if st.button("Use this text in Text-to-Speech"):
                st.session_state.tts_text = transcription_result
                st.rerun()
    
    # Tab 2: Text-to-Speech
    with tab2:
        st.header("Speech Synthesis")
        st.markdown("""
        Convert text to lifelike speech using Azure's neural text-to-speech voices.
        Choose from various languages and voice options.
        """)
        
        # Organize voices by language
        voice_options = get_voice_options()
        
        # Voice selection
        selected_language = st.selectbox("Select language", list(voice_options.keys()))
        selected_voice = st.selectbox(
            "Select voice", 
            voice_options[selected_language], 
            format_func=lambda x: x.split('-')[-1].replace("Neural", " (Neural)")
        )
        
        # Show voice details
        st.info(f"Selected voice: {selected_voice}")
        
        # Text input
        default_text = st.session_state.get("tts_text", "Hello, welcome to Azure Speech Services. I am a neural text-to-speech voice.")
        text_input = st.text_area("Enter text to synthesize:", value=default_text, height=150)
        
        # Synthesize button
        if st.button("Synthesize Speech"):
            if text_input:
                with st.spinner("Synthesizing speech..."):
                    audio_bytes = synthesize_speech(text_input, selected_voice)
                
                if audio_bytes:
                    # Display audio player
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Provide download link
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"speech_synthesis_{timestamp}.wav"
                    st.markdown(get_audio_download_link(audio_bytes, filename, "Download Audio"), unsafe_allow_html=True)
                    
                    # Show SSML example
                    with st.expander("View SSML Example"):
                        ssml = f"""
                        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
                            <voice name="{selected_voice}">
                                {text_input}
                            </voice>
                        </speak>
                        """
                        st.code(ssml, language="xml")
            else:
                st.warning("Please enter some text to synthesize.")
        
        # Voice samples
        with st.expander("Voice Sample Library"):
            st.markdown("Listen to samples of different neural voices:")
            
            sample_col1, sample_col2 = st.columns(2)
            with sample_col1:
                st.markdown("#### English Voices")
                if st.button("Sample: Jenny (US)"):
                    sample_text = "Hello, I'm Jenny, a neural voice from Azure Speech Services."
                    audio_bytes = synthesize_speech(sample_text, "en-US-JennyNeural")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                
                if st.button("Sample: Guy (US)"):
                    sample_text = "Hello, I'm Guy, a neural voice from Azure Speech Services."
                    audio_bytes = synthesize_speech(sample_text, "en-US-GuyNeural")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
            
            with sample_col2:
                st.markdown("#### Other Languages")
                if st.button("Sample: Elvira (Spanish)"):
                    sample_text = "Hola, soy Elvira, una voz neural de Azure Speech Services."
                    audio_bytes = synthesize_speech(sample_text, "es-ES-ElviraNeural")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                
                if st.button("Sample: Henri (French)"):
                    sample_text = "Bonjour, je suis Henri, une voix neurale d'Azure Speech Services."
                    audio_bytes = synthesize_speech(sample_text, "fr-FR-HenriNeural")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
    
    # Tab 3: Speech Translation
    with tab3:
        st.header("Speech Translation")
        st.markdown("""
        Translate spoken language in real-time. Speak into your microphone and have your speech
        translated into multiple languages simultaneously.
        """)
        
        # Source language selection
        source_languages = get_recognition_languages()
        selected_source = st.selectbox(
            "Select source language",
            source_languages,
            format_func=get_language_name,
            key="source_language"
        )
        
        # Target language selection (multiple)
        target_languages = st.multiselect(
            "Select target languages (up to 5)",
            [lang for lang in source_languages if lang != selected_source],
            format_func=get_language_name,
            default=[lang for lang in ["en-US", "es-ES", "fr-FR", "de-DE", "it-IT"] if lang != selected_source][:2]
        )
        
        # Limit to 5 target languages
        if len(target_languages) > 5:
            st.warning("You can select up to 5 target languages. Only the first 5 will be used.")
            target_languages = target_languages[:5]
        
        # Start translation
        if st.button("Start Translation"):
            if target_languages:
                with st.spinner("Listening for speech to translate..."):
                    source_text, translations = translate_speech(target_languages)
                
                if source_text and translations:
                    # Display results
                    st.subheader("Translation Results")
                    
                    # Source text
                    st.markdown(f"**Source ({get_language_name(selected_source)})**: {source_text}")
                    
                    # Translations
                    for lang_code, translation in translations.items():
                        st.markdown(f"**{get_language_name(lang_code)}**: {translation}")
                    
                    # Option to synthesize translations
                    with st.expander("Synthesize translations"):
                        for lang_code, translation in translations.items():
                            st.markdown(f"### {get_language_name(lang_code)}")
                            
                            # Find appropriate voice for this language
                            voice_options_flat = [voice for voices in voice_options.values() for voice in voices]
                            matching_voices = [v for v in voice_options_flat if lang_code in v]
                            
                            if matching_voices:
                                voice = matching_voices[0]  # Use first matching voice
                                if st.button(f"Speak in {voice.split('-')[-1].replace('Neural', '')}", key=f"speak_{lang_code}"):
                                    audio_bytes = synthesize_speech(translation, voice)
                                    if audio_bytes:
                                        st.audio(audio_bytes, format="audio/wav")
                            else:
                                st.warning(f"No voice available for {get_language_name(lang_code)}")
            else:
                st.error("Please select at least one target language.")
    
    # Tab 4: About Speech Services
    with tab4:
        st.header("About Azure Speech Services")
        
        # Overview section
        st.subheader("Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Azure Speech Services is a cloud-based API service that enables you to build speech-enabled applications.
            It provides a wide range of speech capabilities, including speech-to-text, text-to-speech, and speech translation.
            
            **Key capabilities include:**
            
            - **Speech-to-Text**: Convert spoken audio to text with high accuracy
            - **Text-to-Speech**: Generate natural-sounding speech from text
            - **Speech Translation**: Real-time translation of spoken content
            - **Custom Speech**: Create custom speech recognition models
            - **Custom Voice**: Create custom neural voice fonts
            - **Speaker Recognition**: Identify and verify speakers by their voice
            - **Intent Recognition**: Identify user intents from spoken language
            """)

        
        # Use cases
        st.subheader("Common Use Cases")
        use_case_col1, use_case_col2 = st.columns(2)
        
        with use_case_col1:
            st.markdown("""
            **Customer Service**
            - Voice-enabled chatbots and virtual assistants
            - Call center transcription and analytics
            - Interactive voice response (IVR) systems
            
            **Accessibility**
            - Real-time captioning for meetings and events
            - Audio content transcription
            - Reading assistance for visually impaired users
            """)
        
        with use_case_col2:
            st.markdown("""
            **Content Creation**
            - Automated voiceovers for videos
            - Audiobook production
            - Podcast transcription
            
            **Multilingual Experiences**
            - Real-time interpretation for international meetings
            - Multilingual customer support
            - Language learning applications
            """)
        
        # Key features
        st.subheader("Advantages")
        
        st.markdown("""
        - **High Quality**: Neural voices and recognition models deliver natural-sounding speech and accurate transcription
        - **Multilingual Support**: Supports 100+ languages for recognition and 75+ neural voices across 45+ languages
        - **Customization**: Custom models for specific domains, terminology, and acoustics
        - **Integration**: Easy integration with other Azure services like LUIS and Bot Service
        - **Scalability**: Handles varying workloads from small applications to enterprise systems
        - **Security**: Enterprise-grade security and compliance
        """)
        
        # Additional resources
        st.subheader("Additional Resources")
        
        st.markdown("""
        - [Azure Speech Service Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/)
        - [Speech SDK Reference](https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/?view=azure-python)
        - [Sample Applications](https://github.com/Azure-Samples/cognitive-services-speech-sdk)
        - [Pricing Information](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/)
        """)

if __name__ == "__main__":
    show_speech_services()