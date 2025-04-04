import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path so we can import from subfolders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Azure AI Services Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar with service selection
    st.sidebar.title("Azure AI Services")
    st.sidebar.image("https://azure.microsoft.com/svghandler/cognitive-services/?width=600&height=315", width=250)
    
    # Service selection
    service = st.sidebar.selectbox(
        "Select a service to explore",
        ["Home", "Text Analytics", "Question Answering", "Speech Services", "Computer Vision", "Form Recognizer", "Language Understanding"]
    )
    
    # Display selected service content
    if service == "Home":
        show_home()
    elif service == "Text Analytics":
        # Import and show Text Analytics page
        from text_analytics.text_analytics_app import show_text_analytics
        show_text_analytics()
    elif service == "Question Answering":
        # Import and show Question Answering page
        from text_analytics.question_answering_app import show_question_answering
        show_question_answering()
    elif service == "Speech Services":
        st.title("Speech Services")
        st.info("Speech Services features coming soon!")
    elif service == "Computer Vision":
        st.title("Computer Vision")
        st.info("Computer Vision features coming soon!")
    elif service == "Form Recognizer":
        st.title("Form Recognizer")
        st.info("Form Recognizer features coming soon!")
    elif service == "Language Understanding":
        st.title("Language Understanding")
        st.info("Language Understanding features coming soon!")

def show_home():
    st.title("Azure AI Services Explorer")
    st.subheader("Your interactive guide to Azure AI capabilities")
    
    st.markdown("""
    Welcome to the Azure AI Services Explorer! This application demonstrates the power and versatility 
    of Azure's AI and Cognitive Services. Use the sidebar to navigate between different services and explore 
    their capabilities.
    
    ### Available Services:
    
    - **Text Analytics**: Analyze text for sentiment, extract key phrases, detect language, and identify entities
    - **Question Answering**: Get answers from text documents using natural language questions
    - **Speech Services**: Convert speech to text, text to speech, and perform speech translation
    - **Computer Vision**: Analyze images, detect objects, and extract text from images
    - **Form Recognizer**: Extract information from documents and forms
    - **Language Understanding**: Build natural language understanding into your applications
    
    
    ### Getting Started
    
    1. Select a service from the sidebar
    2. Explore the service's capabilities through interactive demos
    3. View the sample code to implement these features in your own applications
    
    This tool is perfect for Azure AI-102 exam preparation and for showcasing Azure AI capabilities to clients.
    """)
    
    # Display key features in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Explore")
        st.markdown("Interactive demonstrations of Azure AI services capabilities")
    
    with col2:
        st.markdown("### 💡 Learn")
        st.markdown("Understand how these services can be applied to real-world scenarios")
    
 
if __name__ == "__main__":
    main()