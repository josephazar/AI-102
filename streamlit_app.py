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
        ["Home", "Text Analytics", "Question Answering", "Conversational Language Understanding", 
         "Computer Vision", "Florence-2 Vision", "Document Intelligence", "AI Search", "Content Understanding", "AI Agents", "Speech Services"]
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
    elif service == "Conversational Language Understanding":
        # Import and show NLU page
        from NLU.nlu_app import show_nlu
        show_nlu()
    elif service == "Computer Vision":
        # Import and show Computer Vision page
        from computer_vision.cv_app import show_computer_vision
        show_computer_vision()
    elif service == "Florence-2 Vision":
        # Import and show Florence-2 page
        from florence.florence_app import show_florence
        show_florence()
    elif service == "Document Intelligence":
        # Import and show Document Intelligence page
        from documents_intelligence.doc_intelligence_app import show_document_intelligence
        show_document_intelligence()
    elif service == "AI Search":
        # Import and show AI Search page
        from ai_search.ai_search_app import show_ai_search
        show_ai_search()
    elif service == "Content Understanding":
        # Import and show Content Understanding page
        from content_understanding.content_understanding_app import show_content_understanding
        show_content_understanding()
    elif service == "AI Agents":
        # Import and show AI Agents page
        from ai_azure_agents.ai_agents_app import show_ai_agents
        show_ai_agents()
    elif service == "Speech Services":
        # Import and show Speech Services page
        from speech.speech_app import show_speech_services
        show_speech_services()

def show_home():
    st.title("Azure AI Services Explorer")
    st.subheader("Your interactive guide to Azure AI capabilities")
    
    # add image foundry-stack-wp.png below title
    st.image("foundry-stack-wp.png", width=1000)
    st.markdown("""
    Welcome to the Azure AI Services Explorer! This application demonstrates the power and versatility 
    of Azure's AI and Cognitive Services. Use the sidebar to navigate between different services and explore 
    their capabilities.
    
    ### Available Services:
    
    - **Text Analytics**: Analyze text for sentiment, extract key phrases, detect language, and identify entities
    - **Question Answering**: Get answers from text documents using natural language questions
    - **Conversational Language Understanding**: Build intelligent bots that understand user intents and extract entities
    - **AI Search**: Create powerful search experiences with vector search, semantic ranking, and relevance tuning
    - **Computer Vision**: Analyze images, detect objects, and extract text from images
    - **Florence-2 Vision**: Explore Microsoft's advanced multi-task vision model with prompt-based interface
    - **Document Intelligence**: Extract information from documents and forms
    - **Content Understanding**: Analyze videos and generate insights, listings, and searchable content
    - **AI Agents**: Build intelligent conversational agents with multi-agent orchestration capabilities
    - **Speech Services**: Convert speech to text, text to speech, and translate spoken language in real-time
    
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