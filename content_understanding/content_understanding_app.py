"""
Streamlit app for Azure Content Understanding - Display Only Mode
"""
import streamlit as st
import os
import time
import glob
import random
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import json
import docx
from io import BytesIO

# Import from our video_processor module for chat functionality
from content_understanding.video_processor import chat_with_video

# Import constants from content_understanding_utils
from content_understanding.content_understanding_utils import (
    get_jpg_files,
    IMAGES_DIR,
    RESULTS_DIR,
    DOCUMENTS_DIR,
    SCRIPT_DIR
)

def show_content_understanding():
    """Main function to display the Content Understanding demo UI."""
    st.title("Azure Content Understanding for Real Estate")
    
 
    # Create tabs for different sections
    tabs = st.tabs([
        "Overview", 
        "Understanding Schemas",
        "Sample Video", 
        "Extracted Keyframes", 
        "Generated Listing", 
        "Chat with Video"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        show_overview()
    
    # Tab 2: Understanding Schemas
    with tabs[1]:
        show_schemas()
        
    # Tab 3: Sample Video
    with tabs[2]:
        show_sample_video()
    
    # Tab 4: Extracted Keyframes
    with tabs[3]:
        show_keyframes()
    
    # Tab 5: Generated Listing
    with tabs[4]:
        show_generated_listing()
    
    # Tab 6: Chat with Video
    with tabs[5]:
        chat_with_video_ui()


def show_overview():
    """Show the overview section with explanation of the demo."""
    st.header("Azure Content Understanding")
    
    # Display the azure_cu.jpg image if it exists
    image_path = os.path.join(SCRIPT_DIR, "azure_cu.jpg")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"Image not found at {image_path}")
    
    st.markdown("""
    ## Video Analysis with Azure Content Understanding
    
    This demo showcases how Azure Content Understanding can analyze real estate videos to:
    
    1. **Extract Detailed Scene Descriptions**: Automatically identify and describe different areas and features of a property
    2. **Generate High-Quality Keyframes**: Capture the most informative visual moments in the video
    3. **Create Professional Listings**: Generate marketing materials with AI-powered descriptions
    4. **Enable Semantic Search**: Allow users to query the video content using natural language
    
    The process uses several Azure AI services working together:
    - **Azure Content Understanding**: For video analysis and feature extraction
    - **Azure OpenAI**: For natural language generation and understanding
    - **Azure AI Search**: For vector-based semantic search
    
    This demo shows the pre-processed results - the actual processing takes several minutes to complete.
    """)


def show_schemas():
    """Show example schemas and explain their purpose to business users."""
    st.header("Understanding Content Understanding Schemas")
    
    st.markdown("""
    ## What is a Schema?
    
    A schema is a structured template that defines how Azure Content Understanding should analyze your content. Think of it as instructions for the AI system about what to look for and what information to extract.
    
    ### Why Schemas Matter for Business
    
    - **Customized Analysis**: Tailor the AI to extract exactly the information that matters to your business
    - **Consistent Results**: Ensure that all content is analyzed in the same way
    - **Domain-Specific Insights**: Create schemas specific to your industry (real estate, healthcare, retail, etc.)
    - **Flexible Application**: Apply the same schema to videos, images, or documents
    
    Below are two example schemas that illustrate how you can define different analysis goals:
    """)
    
    # Display the Real Estate schema in a user-friendly format
    st.subheader("Example 1: Real Estate Property Analysis Schema")
    
    st.markdown("""
    This schema tells the AI to analyze real estate videos and extract information suitable for creating property listings.
    """)
    
    # Create a more business-friendly table representation of the real estate schema
    real_estate_schema = {
        "Schema Purpose": "Analyze real estate videos to create property listings",
        "Content Type": "Video",
        "Primary Task": "Extract property features and create compelling descriptions",
        "Key Information to Extract": [
            "Number of bedrooms and bathrooms",
            "Total square footage",
            "Location details",
            "Architectural style",
            "Luxury elements (appliances, finishes, materials)",
            "Unique selling points",
            "Lifestyle benefits",
            "Proximity to amenities"
        ],
        "Output Style": "Professional real estate listing with sophisticated tone",
        "Supported Languages": "English, Spanish, French, Hindi, Italian, Japanese, Korean, Portuguese, Chinese"
    }
    
    # Display the schema components in a more visual way
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Schema Components")
        for key in real_estate_schema:
            if key != "Key Information to Extract":
                st.markdown(f"**{key}**")
    
    with col2:
        st.markdown("#### Configuration Values")
        for key, value in real_estate_schema.items():
            if key != "Key Information to Extract":
                if isinstance(value, list):
                    st.markdown(", ".join(value))
                else:
                    st.markdown(value)
    
    st.markdown("#### Key Information to Extract")
    for item in real_estate_schema["Key Information to Extract"]:
        st.markdown(f"- {item}")
    
    # Display the second schema - Audio Transcription
    st.subheader("Example 2: Audio Transcription Schema")
    
    st.markdown("""
    This simpler schema focuses on transcribing audio conversations. It demonstrates how schemas can be tailored for different content types and business needs.
    """)
    
    # Create a business-friendly representation of the audio schema
    audio_schema = {
        "Schema Purpose": "Transcribe spoken conversations",
        "Content Type": "Audio",
        "Primary Task": "Convert speech to text",
        "Key Information to Extract": [
            "Spoken words",
            "Speaker identification",
            "Conversation flow"
        ],
        "Output Style": "Accurate text transcription",
        "Supported Languages": "English (US)"
    }
    
    # Display the schema components
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Schema Components")
        for key in audio_schema:
            if key != "Key Information to Extract":
                st.markdown(f"**{key}**")
    
    with col2:
        st.markdown("#### Configuration Values")
        for key, value in audio_schema.items():
            if key != "Key Information to Extract":
                if isinstance(value, list):
                    st.markdown(", ".join(value))
                else:
                    st.markdown(value)
    
    st.markdown("#### Key Information to Extract")
    for item in audio_schema["Key Information to Extract"]:
        st.markdown(f"- {item}")
    
    # Business benefits of using schemas
    st.subheader("How Schemas Drive Business Value")
    
    benefit_cols = st.columns(3)
    
    with benefit_cols[0]:
        st.markdown("#### Efficiency")
        st.markdown("""
        - Automate content analysis
        - Reduce manual review time
        - Process large volumes of content
        """)
    
    with benefit_cols[1]:
        st.markdown("#### Consistency")
        st.markdown("""
        - Standardize information extraction
        - Create uniform outputs
        - Ensure comprehensive analysis
        """)
    
    with benefit_cols[2]:
        st.markdown("#### Adaptability")
        st.markdown("""
        - Easily modify for different use cases
        - Create multiple specialized schemas
        - Apply across different content types
        """)
    
    st.markdown("""
    ### Getting Started with Schemas
    
    To create your own schema:
    
    1. **Identify your business goals**: What information do you need from your content?
    2. **Define the key fields**: What specific data points should the AI extract?
    3. **Set analysis parameters**: Configure languages, detail level, and output formats
    4. **Test and refine**: Apply to sample content and adjust as needed
    
    No coding required – schemas can be created through user-friendly interfaces and saved as JSON templates for reuse.
    """)

def show_sample_video():
    """Show the sample video."""
    st.header("Sample Real Estate Video")
    
    video_path = os.path.join(DOCUMENTS_DIR, "paris.mp4")
    if os.path.exists(video_path):
        st.video(video_path)
        
        # Display some information about the video
        st.subheader("Video Information")
        try:
            import cv2
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            st.write(f"**Duration:** {int(duration // 60)} minutes {int(duration % 60)} seconds")
            st.write(f"**Frame Rate:** {fps:.2f} fps")
            st.write(f"**Resolution:** {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            video.release()
        except Exception as e:
            st.write("**Video File:** paris.mp4")
    else:
        st.error(f"Video not found at {video_path}")
        st.write("Please ensure the video file 'paris.mp4' is in the 'video' directory.")

def show_keyframes():
    """Show extracted keyframes from the video."""
    st.header("Extracted Keyframes")
    
    # Get all keyframe files from the frames directory
    keyframe_files = glob.glob(os.path.join(IMAGES_DIR, "keyFrame*.jpg"))
    
    if keyframe_files:
        st.write(f"**Total Keyframes Extracted:** {len(keyframe_files)}")
        
        # Display a selection of random keyframes
        st.subheader("Sample Keyframes")
        
        # Select 12-15 random keyframes
        num_samples = min(15, len(keyframe_files))
        selected_frames = random.sample(keyframe_files, num_samples)
        
        # Create a grid of 3 columns
        cols = st.columns(3)
        
        # Display each selected frame
        for i, frame_path in enumerate(selected_frames):
            with cols[i % 3]:
                st.image(frame_path, use_container_width=True, caption=os.path.basename(frame_path))
    else:
        st.warning(f"No keyframes found in {IMAGES_DIR}")
        st.write("Please ensure keyframes have been extracted to the 'frames' directory.")

def show_generated_listing():
    """Show the generated real estate listing."""
    st.header("Generated Real Estate Listing")
    
    # Look for the docx file
    docx_file = os.path.join(RESULTS_DIR, "real_estate_listing_paris.docx")
    
    if os.path.exists(docx_file):
        # Extract and display the content
        try:
            doc = docx.Document(docx_file)
            
            # Display the document content
            for para in doc.paragraphs:
                if para.style.name.startswith('Heading'):
                    st.subheader(para.text)
                else:
                    st.write(para.text)
            
            # Find and display images in the document
            displayed_images = False
            if len(doc.inline_shapes) > 0:
                st.subheader("Property Images")
                displayed_images = True
                
                # Create columns for images
                cols = st.columns(2)
                col_idx = 0
                
                # Try to extract images from the docx
                try:
                    temp_dir = os.path.join(RESULTS_DIR, "temp_images")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Alternative: Just display keyframe images from the frames directory
                    keyframe_files = glob.glob(os.path.join(IMAGES_DIR, "keyFrame*.jpg"))
                    selected_frames = random.sample(keyframe_files, min(6, len(keyframe_files)))
                    
                    for i, frame_path in enumerate(selected_frames):
                        with cols[i % 2]:
                            st.image(frame_path, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error displaying images: {str(e)}")
            
            if not displayed_images:
                # Show random keyframes instead
                st.subheader("Property Images")
                keyframe_files = glob.glob(os.path.join(IMAGES_DIR, "keyFrame*.jpg"))
                if keyframe_files:
                    selected_frames = random.sample(keyframe_files, min(6, len(keyframe_files)))
                    
                    cols = st.columns(2)
                    for i, frame_path in enumerate(selected_frames):
                        with cols[i % 2]:
                            st.image(frame_path, use_container_width=True)
            
            # Provide download button for the docx
            with open(docx_file, "rb") as file:
                st.download_button(
                    label="Download Full Listing Document",
                    data=file,
                    file_name=os.path.basename(docx_file),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        
        except Exception as e:
            st.error(f"Error reading document: {str(e)}")
            st.write("Unable to display document content. You can still download the document using the button below.")
            
            # Provide download button anyway
            with open(docx_file, "rb") as file:
                st.download_button(
                    label="Download Listing Document",
                    data=file,
                    file_name=os.path.basename(docx_file),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    else:
        st.warning(f"Listing document not found at {docx_file}")
        st.write("Please ensure the listing has been generated and saved to the 'results' directory.")

def load_vector_store():
    """Load the vector store from a file if it exists."""
    try:
        # Try to load from session state first
        if 'vector_store' in st.session_state:
            return st.session_state['vector_store']
        
        # Otherwise, check for pre-processed vector store
        # This is a simplified placeholder - you'll need to implement actual vector store loading
        from content_understanding.content_understanding_utils import (
            load_into_index, 
            AZURE_SEARCH_INDEX_NAME
        )
        
        if 'search_loaded' not in st.session_state:
            from content_understanding.content_understanding_utils import AzureSearch
            # Placeholder for loading vector store - modify based on your actual storage mechanism
            st.session_state['vector_store'] = "PLACEHOLDER - REPLACE WITH ACTUAL VECTOR STORE"
            st.session_state['search_loaded'] = True
            
        return st.session_state['vector_store']
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def chat_with_video_ui():
    """UI for chatting with video content."""
    st.header("Chat with Video")
    st.markdown("""
    Ask questions about the property shown in the video and get AI-generated answers based on the video content.
    
    Example questions:
    - What kind of rooms are shown in the video?
    - Is there a balcony or terrace?
    - What amenities does the property have?
    - Describe the kitchen in the property.
    - What's the view like from this property?
    """)
    
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
    
    # Input for new question
    user_question = st.chat_input("Ask a question about the video")
    
    if user_question:
        # Display user question
        with st.chat_message("user"):
            st.write(user_question)
        
        # Add to history
        st.session_state['chat_history'].append({
            'role': 'user',
            'content': user_question
        })
        
        # Get vector store
        vector_store = load_vector_store()
        
        if vector_store:
            # Generate response
            with st.spinner("Analyzing the video content..."):
                # If we don't have a real vector store, simulate responses for demo purposes
                if isinstance(vector_store, str) and vector_store.startswith("PLACEHOLDER"):
                    # Simulation mode - predefined responses
                    if "balcony" in user_question.lower() or "terrace" in user_question.lower():
                        response = "Yes, the property features a beautiful terrace with stunning views of Paris, including the iconic Eiffel Tower in the distance. The terrace is spacious and appears to have outdoor furniture, making it perfect for entertaining or relaxing outdoors."
                    elif "kitchen" in user_question.lower():
                        response = "The kitchen in this property is modern and well-appointed. It features sleek cabinetry, high-end appliances, and a clean design aesthetic. The kitchen appears to be open-concept, connecting to the living area, which creates a nice flow for entertaining."
                    elif "view" in user_question.lower():
                        response = "The property offers breathtaking views of Paris, with the Eiffel Tower visible from several windows and the terrace. The cityscape view is particularly impressive at different times of day, showcasing Paris's iconic architecture and urban landscape."
                    elif "room" in user_question.lower() or "bedroom" in user_question.lower():
                        response = "The video shows several rooms including a spacious living room with modern furnishings, what appears to be at least one bedroom with elegant decor, a stylish bathroom, and a modern kitchen. The rooms feature large windows that flood the space with natural light and offer views of the city."
                    elif "amenities" in user_question.lower() or "features" in user_question.lower():
                        response = "The property includes several notable amenities: a spacious terrace with city views, modern kitchen with high-end appliances, elegant bathroom fixtures, large windows throughout, hardwood flooring, and what appears to be built-in storage solutions. The location also seems to be a key amenity, with proximity to Parisian landmarks."
                    else:
                        response = "Based on the video, this appears to be a luxury apartment or condo in Paris with modern design elements. It features spacious rooms with large windows, elegant furnishings, and a terrace with views of the city including the Eiffel Tower. The property has a sophisticated urban aesthetic that combines contemporary design with classic Parisian architectural elements."
                else:
                    # Use the actual chat_with_video function
                    response = chat_with_video(user_question, vector_store)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response)
                
                # Add to history
                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'content': response
                })
        else:
            with st.chat_message("assistant"):
                st.write("I'm sorry, but I can't access the video content right now. The vector store with the video analysis data isn't available.")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()

if __name__ == "__main__":
    show_content_understanding()