import streamlit as st
import os
import time
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Azure Computer Vision settings
ENDPOINT = os.getenv('COMPUTER_VISION_ENDPOINT')
KEY = os.getenv('COMPUTER_VISION_KEY')
REGION = os.getenv('COMPUTER_VISION_REGION')

# Define color scheme for the app
COLORS = {
    'primary': '#0078D4',    # Azure blue
    'secondary': '#50E6FF',  # Light blue
    'objects': '#F7630C',    # Orange
    'people': '#0078D4',     # Blue
    'text': '#107C10',       # Green
    'caption': '#6264A7',    # Purple
    'tags': '#C239B3',       # Pink
    'accent': '#FFB900',     # Gold
    'error': '#D13438'       # Red
}

# ----- Azure Computer Vision Core Functions -----

def analyze_image(image_data, features="caption,read,tags,objects,people,denseCaptions"):
    """
    Analyze an image using Azure Computer Vision API.
    
    Args:
        image_data (bytes): Image data
        features (str): Comma-separated list of features to analyze
    
    Returns:
        dict: Analysis results
    """
    if not ENDPOINT or not KEY:
        st.error("Missing Azure Computer Vision credentials. Please check your .env file.")
        return None
    
    # API endpoint for image analysis
    analyze_url = f"{ENDPOINT}computervision/imageanalysis:analyze"
    
    # Request headers
    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Content-Type": "application/octet-stream"
    }
    
    # Request parameters
    params = {
        "features": features,
        "model-version": "latest",
        "language": "en",
        "api-version": "2023-10-01"
    }
    
    # Make request to Computer Vision API
    try:
        with st.spinner("Analyzing image..."):
            response = requests.post(
                analyze_url,
                headers=headers,
                params=params,
                data=image_data
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e}")
        try:
            error_json = response.json()
            st.error(f"API Error: {error_json.get('error', {}).get('message', 'Unknown error')}")
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def analyze_image_url(image_url, features="caption,read,tags,objects,people"):
    """
    Analyze an image from URL using Azure Computer Vision API.
    
    Args:
        image_url (str): URL to image
        features (str): Comma-separated list of features to analyze
    
    Returns:
        dict: Analysis results
    """
    if not ENDPOINT or not KEY:
        st.error("Missing Azure Computer Vision credentials. Please check your .env file.")
        return None
    
    # API endpoint for image analysis
    analyze_url = f"{ENDPOINT}computervision/imageanalysis:analyze"
    
    # Request headers
    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Content-Type": "application/json"
    }
    
    # Request parameters
    params = {
        "features": features,
        "model-version": "latest",
        "language": "en",
        "api-version": "2023-10-01"
    }
    
    # Request body
    data = {
        "url": image_url
    }
    
    # Make request to Computer Vision API
    try:
        with st.spinner("Analyzing image..."):
            response = requests.post(
                analyze_url,
                headers=headers,
                params=params,
                json=data
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e}")
        try:
            error_json = response.json()
            st.error(f"API Error: {error_json.get('error', {}).get('message', 'Unknown error')}")
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def draw_bounding_boxes(image, analysis_result):
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    # Define colors (assuming COLORS is defined elsewhere, e.g., in a config)
    COLORS = {'people': 'blue', 'objects': 'orange'}  # Adjust as per your app

    if 'objectsResult' in analysis_result and 'values' in analysis_result['objectsResult']:
        for obj in analysis_result['objectsResult']['values']:
            bbox = obj.get('boundingBox', {})
            if all(key in bbox for key in ['x', 'y', 'w', 'h']):
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                # Extract tag info
                if obj.get('tags'):
                    tag = obj['tags'][0]  # Primary tag
                    obj_name = tag.get('name', 'object').lower()
                    confidence = tag.get('confidence', 0)
                else:
                    obj_name = 'object'
                    confidence = 0

                # Assign color based on whether it's a person
                color = COLORS['people'] if obj_name == 'person' else COLORS['objects']

                # Draw rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                # Draw label with confidence
                label = f"{obj_name} ({confidence:.2f})"
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                draw.rectangle([x, y - text_h - 2, x + text_w, y], fill=color)
                draw.text((x, y - text_h - 2), label, fill="white", font=font)

    return image_with_boxes


def draw_text_areas(image, analysis_result):
    """
    Draw text areas on the image.
    
    Args:
        image (PIL.Image): Original image
        analysis_result (dict): Analysis results from Computer Vision API
    
    Returns:
        PIL.Image: Image with text areas
    """
    # Create a copy of the image to draw on
    image_with_text = image.copy()
    draw = ImageDraw.Draw(image_with_text)
    
    # Draw text areas if available
    if 'readResult' in analysis_result and 'blocks' in analysis_result['readResult']:
        for block in analysis_result['readResult']['blocks']:
            for line in block.get('lines', []):
                # Get bounding polygon
                polygon = line.get('boundingPolygon', [])
                if len(polygon) >= 4:
                    # Convert to flat list of coordinates
                    points = [(point['x'], point['y']) for point in polygon]
                    
                    # Draw polygon
                    draw.polygon(points, outline=COLORS['text'], width=2)
    
    return image_with_text

def get_all_text(analysis_result):
    """
    Extract all text from OCR results.
    
    Args:
        analysis_result (dict): Analysis results from Computer Vision API
    
    Returns:
        list: Extracted text lines
    """
    text_lines = []
    
    if 'readResult' in analysis_result and 'blocks' in analysis_result['readResult']:
        for block in analysis_result['readResult']['blocks']:
            for line in block.get('lines', []):
                text_lines.append(line.get('text', ''))
    
    return text_lines

def create_tag_chart(tags):
    """
    Create a bar chart for tags and their confidence scores.
    
    Args:
        tags (list): List of tag objects with 'name' and 'confidence'
    
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    # Prepare data
    tags_df = pd.DataFrame(tags)
    
    # Sort by confidence score
    tags_df = tags_df.sort_values('confidence', ascending=False).head(15)
    
    # Create chart
    fig = px.bar(
        tags_df,
        x='confidence',
        y='name',
        orientation='h',
        labels={"confidence": "Confidence Score", "name": "Tag"},
        title="Top Tags by Confidence",
        color_discrete_sequence=[COLORS['tags']]
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        font=dict(family="Arial, sans-serif"),
        title_font=dict(family="Arial, sans-serif", size=16)
    )
    
    return fig

def create_object_chart(objects):
    """
    Create a pie chart for detected objects.
    
    Args:
        objects (list): List of detected objects
    
    Returns:
        plotly.graph_objects.Figure: Pie chart
    """
    # Safely extract object names
    object_names = []
    for obj in objects:
        # Get object name from tags if available
        if 'tags' in obj and len(obj['tags']) > 0 and 'name' in obj['tags'][0]:
            object_names.append(obj['tags'][0]['name'])
        else:
            object_names.append("Unknown Object")
    
    # Count objects by name
    object_counts = Counter(object_names)
    
    # Create chart
    if len(object_counts) > 1:
        # Create pie chart for multiple objects
        fig = px.pie(
            values=list(object_counts.values()),
            names=list(object_counts.keys()),
            title="Detected Objects",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=300,
            font=dict(family="Arial, sans-serif"),
            title_font=dict(family="Arial, sans-serif", size=16)
        )
        
        # Update traces
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    else:
        # Create gauge chart for single object type
        name = list(object_counts.keys())[0]
        count = list(object_counts.values())[0]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=count,
            title={"text": f"Detected: {name}", "font": {"size": 16}},
            gauge={"axis": {"range": [0, max(5, count * 1.5)]}, "bar": {"color": COLORS['objects']}},
            number={"font": {"size": 40}}
        ))
        
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=300,
            font=dict(family="Arial, sans-serif")
        )
        
        return fig

def create_confidence_chart(result):
    confidence_scores = []
    if 'captionResult' in result and 'confidence' in result['captionResult']:
        confidence_scores.append(result['captionResult']['confidence'])
    if 'tagsResult' in result and 'values' in result['tagsResult']:
        for tag in result['tagsResult']['values']:
            confidence_scores.append(tag.get('confidence', 0))
    if 'objectsResult' in result and 'values' in result['objectsResult']:
        for obj in result['objectsResult']['values']:
            if obj.get('tags'):
                confidence_scores.append(obj['tags'][0].get('confidence', 0))

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_confidence * 100,
        title={"text": "Overall Confidence", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#0078D4"},
            "steps": [
                {"range": [0, 50], "color": "#FFCCCB"},
                {"range": [50, 75], "color": "#FFFFCC"},
                {"range": [75, 100], "color": "#CCFFCC"}
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 75}
        },
        number={"suffix": "%", "font": {"size": 40}}
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=250, font=dict(family="Arial, sans-serif"))
    return fig
# ----- Streamlit UI Components -----

def set_custom_css():
    """Set custom CSS styles for the app."""
    st.markdown("""
    <style>
    /* Basic styling to ensure consistency */
    .feature-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .feature-header {
        color: #0078D4;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #0078D4;
        padding-bottom: 5px;
    }
    
    .caption-box {
        background-color: #f0e7f7;
        border-left: 5px solid #6264A7;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
    }
    
    .tag-item {
        display: inline-block;
        background-color: #f5ebf5;
        color: #C239B3;
        border: 1px solid #C239B3;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 14px;
    }
    
    .text-box {
        background-color: #f0f9ff;
        border: 1px solid #cfe2ff;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .text-line {
        border-bottom: 1px solid #e9ecef;
        padding: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    """Display the hero header section of the app."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0078D4 0%, #50E6FF 100%); color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px; text-align: center;">
            <h1 style="font-size: 36px; font-weight: bold; margin-bottom: 15px;">Azure Computer Vision Explorer</h1>
            <p style="font-size: 18px; margin-bottom: 20px;">Unlock the potential of visual content with Azure's AI-powered image analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Use native Streamlit components for stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Visual Features", value="5+")
    
    with col2:
        st.metric(label="Accuracy", value="99%")
    
    with col3:
        st.metric(label="Objects Recognized", value="100K+")
    
    with col4:
        st.metric(label="Languages", value="25+")


def show_image_upload_section():
    """
    Show image upload section and handle uploads.
    
    Returns:
        tuple: (image_data, image, source_type)
    """
    # Initialize session state for storing image data
    if 'image_data' not in st.session_state:
        st.session_state.image_data = None
        st.session_state.image = None
        st.session_state.source_type = None
        st.session_state.selected_file = None
        st.session_state.selected_url = None
        st.session_state.selected_sample = None
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🔗 Image URL", "📷 Sample Images"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "gif"], key="file_uploader")
        if uploaded_file is not None:
            # Update session state only if uploaded file changes
            if st.session_state.selected_file != uploaded_file.name:
                st.session_state.image_data = uploaded_file.getvalue()
                st.session_state.image = Image.open(BytesIO(st.session_state.image_data))
                st.session_state.source_type = "upload"
                st.session_state.selected_file = uploaded_file.name
                # Reset other selections
                st.session_state.selected_url = None
                st.session_state.selected_sample = None
            
            st.image(st.session_state.image, caption="Uploaded Image", use_container_width=True)
    
    with tab2:
        image_url = st.text_input("Enter image URL", key="url_input")
        if image_url:
            # Update session state only if URL changes
            if st.session_state.selected_url != image_url:
                try:
                    with st.spinner("Loading image..."):
                        response = requests.get(image_url)
                        response.raise_for_status()
                        st.session_state.image_data = response.content
                        st.session_state.image = Image.open(BytesIO(st.session_state.image_data))
                        st.session_state.source_type = "url"
                        st.session_state.selected_url = image_url
                        # Reset other selections
                        st.session_state.selected_file = None
                        st.session_state.selected_sample = None
                except Exception as e:
                    st.error(f"Error loading image from URL: {str(e)}")
            
            if st.session_state.source_type == "url":
                st.image(st.session_state.image, caption="Image from URL", use_container_width=True)
    
    with tab3:
        # Get sample images from both possible locations
        sample_paths = []
        data_folders = [
            os.path.join("computer_vision", "data"),
            "data"
        ]
        
        for folder in data_folders:
            if os.path.exists(folder):
                sample_paths.extend(glob.glob(os.path.join(folder, "*.jpg")))
                sample_paths.extend(glob.glob(os.path.join(folder, "*.jpeg")))
                sample_paths.extend(glob.glob(os.path.join(folder, "*.png")))
        
        if sample_paths:
            # Create a grid layout for sample images
            cols = st.columns(3)
            for i, sample_path in enumerate(sample_paths):
                with cols[i % 3]:
                    sample_img = Image.open(sample_path)
                    # Resize for thumbnail
                    sample_img.thumbnail((150, 150))
                    st.image(sample_img, caption=os.path.basename(sample_path))
                    if st.button(f"Select", key=f"sample_{i}"):
                        # Update session state only if sample changes
                        if st.session_state.selected_sample != sample_path:
                            with open(sample_path, "rb") as f:
                                st.session_state.image_data = f.read()
                            st.session_state.image = Image.open(BytesIO(st.session_state.image_data))
                            st.session_state.source_type = "sample"
                            st.session_state.selected_sample = sample_path
                            # Reset other selections
                            st.session_state.selected_file = None
                            st.session_state.selected_url = None
            
            # Show selected sample image
            if st.session_state.source_type == "sample" and st.session_state.selected_sample:
                st.image(st.session_state.image, caption=f"Selected: {os.path.basename(st.session_state.selected_sample)}", use_container_width=True)
        else:
            st.info("No sample images found. Please run download_images.py first or upload your own image.")
    
    return st.session_state.image_data, st.session_state.image, st.session_state.source_type

def show_analysis_options():
    """
    Show analysis options and get selected features.
    
    Returns:
        str: Comma-separated list of selected features
    """
    st.subheader("Analysis Options")
    
    # Initialize feature state if not already present
    if 'features' not in st.session_state:
        st.session_state.features = {
            'caption': True,
            'objects': True,
            'tags': True,
            'people': True,
            'read': True,
            'domains': True,
            'denseCaptions': True
        }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        caption = st.checkbox("Image Captioning", value=st.session_state.features['caption'], 
                              help="Generate a description of the image content", 
                              key="checkbox_caption")
        st.session_state.features['caption'] = caption
        
        dense_captions = st.checkbox("Dense Captions", value=st.session_state.features['denseCaptions'], 
                              help="Generate detailed captions for regions of the image", 
                              key="checkbox_dense_captions")
        st.session_state.features['denseCaptions'] = dense_captions
        
        objects = st.checkbox("Object Detection", value=st.session_state.features['objects'], 
                             help="Identify objects and their location in the image",
                             key="checkbox_objects")
        st.session_state.features['objects'] = objects
    
    with col2:
        tags = st.checkbox("Tags & Keywords", value=st.session_state.features['tags'], 
                          help="Extract relevant tags and keywords from the image",
                          key="checkbox_tags")
        st.session_state.features['tags'] = tags
        
        people = st.checkbox("People Detection", value=st.session_state.features['people'], 
                            help="Detect people in the image",
                            key="checkbox_people")
        st.session_state.features['people'] = people
    
    with col3:
        read = st.checkbox("Text Recognition (OCR)", value=st.session_state.features['read'], 
                          help="Extract text from the image",
                          key="checkbox_read")
        st.session_state.features['read'] = read
        
        domains = st.checkbox("Domain-Specific Content", value=st.session_state.features['domains'], 
                             help="Identify domain-specific content (celebrities, landmarks)",
                             key="checkbox_domains")
        st.session_state.features['domains'] = domains
    
    # Build features string
    features = []
    if st.session_state.features['caption']:
        features.append("caption")
    if st.session_state.features['tags']:
        features.append("tags")
    if st.session_state.features['objects']:
        features.append("objects")
    if st.session_state.features['people']:
        features.append("people")
    if st.session_state.features['read']:
        features.append("read")
    if st.session_state.features['domains']:
        features.append("smartCrops")
    if st.session_state.features['denseCaptions']:
        features.append("denseCaptions")
    
    return ",".join(features)

def display_dense_captions(result, image):
    """Display dense captions for regions in the image."""
    
    # Check for both possible response formats
    has_captions = False
    captions = []
    
    # Format 1: denseCaptionsResult (as in your original code)
    if 'denseCaptionsResult' in result and 'values' in result['denseCaptionsResult']:
        captions = result['denseCaptionsResult']['values']
        has_captions = True
    
    # Format 2: alternative format in newer API versions
    elif 'captionResult' in result and 'denseCaptionsResult' in result['captionResult']:
        if 'values' in result['captionResult']['denseCaptionsResult']:
            captions = result['captionResult']['denseCaptionsResult']['values']
            has_captions = True
    
    # Format 3: denseCaptions directly in result (another possible format)
    elif 'denseCaptions' in result and 'values' in result['denseCaptions']:
        captions = result['denseCaptions']['values']
        has_captions = True
    
    if has_captions and captions:
        st.markdown("### 📝 Dense Captioning")

        # Draw bounding boxes on the image
        image_with_captions = image.copy()
        draw = ImageDraw.Draw(image_with_captions)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        for caption in captions:
            bbox = caption.get('boundingBox', {})
            if all(key in bbox for key in ['x', 'y', 'w', 'h']):
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                draw.rectangle([x, y, x + w, y + h], outline='green', width=2)
                # Handle different caption text field names
                text = caption.get('text', caption.get('caption', ''))
                confidence = caption.get('confidence', 0)
                label = f"{text} ({confidence:.2f})"
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                draw.rectangle([x, y - text_h - 2, x + text_w, y], fill='green')
                draw.text((x, y - text_h - 2), label, fill="white", font=font)

        st.image(image_with_captions, caption="Image with Dense Captions", use_container_width=True)

        # Display captions in a table
        captions_data = []
        for i, cap in enumerate(captions):
            # Handle different field names
            text = cap.get('text', cap.get('caption', ''))
            confidence = cap.get('confidence', 0)
            captions_data.append({
                "Region": f"Region {i+1}", 
                "Caption": text, 
                "Confidence": f"{confidence:.2f}"
            })
        
        st.dataframe(pd.DataFrame(captions_data), use_container_width=True)
    else:
        st.warning("No dense captions available in the API response. Make sure the 'denseCaptions' feature is enabled in the API.")
        # Log the structure of the response to help debugging
        st.expander("Debug Response Structure").write(str(result.keys()))

def display_caption_results(result):
    """Display image caption analysis results."""
    if 'captionResult' in result and 'text' in result['captionResult']:
        caption = result['captionResult']['text']
        confidence = result['captionResult'].get('confidence', 0)
        
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-header">📝 Image Caption</div>
            <div class="caption-box">
                <h3 style="margin-top:0; font-size:24px;">{caption}</h3>
                <p style="margin-bottom:0;">Confidence: {confidence:.2f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_tags_results(result):
    """Display image tags analysis results."""
    if 'tagsResult' in result and 'values' in result['tagsResult']:
        tags = result['tagsResult']['values']
        
        if tags:
            st.markdown(f"""
            <div class="feature-box">
                <div class="feature-header">🏷️ Tags & Keywords</div>
            """, unsafe_allow_html=True)
            
            # Create tags chart
            chart = create_tag_chart(tags)
            st.plotly_chart(chart, use_container_width=True)
            
            # Display tags as pills
            tag_html = "<div style='margin-top:10px;'>"
            for tag in sorted(tags, key=lambda x: x['confidence'], reverse=True):
                tag_html += f"""<span class="tag-item">{tag['name']} ({tag['confidence']:.2f})</span>"""
            tag_html += "</div>"
            
            st.markdown(tag_html, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def display_objects_results(result, image):
    if 'objectsResult' in result and 'values' in result['objectsResult']:
        objects = result['objectsResult']['values']
        st.markdown("""
            <div class="feature-box">
            <div class="feature-header">🔍 Object Detection</div>
        """, unsafe_allow_html=True)

        # Display image with all bounding boxes
        image_with_boxes = draw_bounding_boxes(image, result)
        st.image(image_with_boxes, caption=f"Detected {len(objects)} Items (People and Objects)", use_container_width=True)

        # Separate people and other objects
        people = [obj for obj in objects if obj.get('tags') and obj['tags'][0]['name'].lower() == 'person']
        other_objects = [obj for obj in objects if not (obj.get('tags') and obj['tags'][0]['name'].lower() == 'person')]

        # Display Detected People
        if people:
            st.markdown(f"<h3 style='color:blue;'>Detected {len(people)} People</h3>", unsafe_allow_html=True)
            people_data = []
            for i, person in enumerate(people):
                confidence = person['tags'][0].get('confidence', 0) if person.get('tags') else 0
                people_data.append({
                    "Person": f"Person {i+1}",
                    "Confidence": f"{confidence:.2f}"
                })
            st.dataframe(pd.DataFrame(people_data), use_container_width=True)

        # Display Detected Other Objects
        if other_objects:
            st.markdown(f"<h3 style='color:orange;'>Detected {len(other_objects)} Other Objects</h3>", unsafe_allow_html=True)
            objects_data = []
            for obj in other_objects:
                obj_name = obj['tags'][0]['name'] if obj.get('tags') else "Unknown"
                confidence = obj['tags'][0].get('confidence', 0) if obj.get('tags') else 0
                objects_data.append({
                    "Object": obj_name,
                    "Confidence": f"{confidence:.2f}"
                })
            st.dataframe(pd.DataFrame(objects_data), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("No objects detected.")


def display_people_results(result, image):
    """Display people detection results."""
    if 'peopleResult' in result and 'values' in result['peopleResult']:
        people = result['peopleResult']['values']
        
        if people:
            st.markdown(f"""
            <div class="feature-box">
                <div class="feature-header">👥 People Detection</div>
            """, unsafe_allow_html=True)
            
            # Display image with bounding boxes
            image_with_people = draw_bounding_boxes(image, result)
            st.image(image_with_people, caption=f"Detected {len(people)} People", use_container_width=True)
            
            # Calculate average confidence
            avg_confidence = sum(person.get('confidence', 0) for person in people) / len(people) if people else 0
            
            # Display stats
            st.markdown(f"""
            <h3 style="text-align:center; color:#0078D4;">{len(people)} People Detected</h3>
            <p style="text-align:center;">Average confidence: {avg_confidence:.2f}</p>
            """, unsafe_allow_html=True)
            
            # Display people details in a table
            with st.expander("People Details"):
                people_data = []
                for i, person in enumerate(people):
                    people_data.append({
                        "Person": f"Person {i+1}",
                        "Confidence": f"{person.get('confidence', 0):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(people_data), use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def display_text_results(result, image):
    """Display OCR (text recognition) results."""
    if 'readResult' in result and 'blocks' in result['readResult']:
        blocks = result['readResult']['blocks']
        
        if blocks:
            # Extract text lines
            text_lines = get_all_text(result)
            
            if text_lines:
                st.markdown(f"""
                <div class="feature-box">
                    <div class="feature-header">📄 Text Recognition (OCR)</div>
                """, unsafe_allow_html=True)
                
                # Display image with text areas
                image_with_text = draw_text_areas(image, result)
                st.image(image_with_text, caption="Detected Text", use_container_width=True)
                
                # Display extracted text
                st.markdown("<h3>Extracted Text:</h3>", unsafe_allow_html=True)
                
                text_html = "<div class='text-box'>"
                for line in text_lines:
                    text_html += f"<div class='text-line'>{line}</div>"
                text_html += "</div>"
                
                st.markdown(text_html, unsafe_allow_html=True)
                
                # Text stats
                st.markdown(f"""
                <div style="margin-top:15px;">
                    <h4>Text Statistics:</h4>
                    <ul>
                        <li>{len(blocks)} text blocks detected</li>
                        <li>{len(text_lines)} lines of text</li>
                        <li>Approximately {sum(len(line.split()) for line in text_lines)} words</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

def display_analysis_overview(result):
    st.markdown("""
        <div class="feature-box">
        <div class="feature-header">📊 Analysis Overview</div>
    """, unsafe_allow_html=True)

    # Confidence chart (updated below in step 4)
    confidence_chart = create_confidence_chart(result)
    st.plotly_chart(confidence_chart, use_container_width=True)

    # Summary stats
    stats = {}
    if 'objectsResult' in result and 'values' in result['objectsResult']:
        objects = result['objectsResult']['values']
        people_count = sum(1 for obj in objects if obj.get('tags') and obj['tags'][0]['name'].lower() == 'person')
        other_count = len(objects) - people_count
        stats["People Detected"] = people_count
        stats["Other Objects Detected"] = other_count

    if 'tagsResult' in result and 'values' in result['tagsResult']:
        stats["Tags Identified"] = len(result['tagsResult']['values'])

    if 'readResult' in result and 'blocks' in result['readResult']:
        text_lines = get_all_text(result)  # Assuming this function exists
        stats["Text Lines Extracted"] = len(text_lines)

    # Display stats
    if stats:
        st.markdown("<div style='display:flex; gap:15px; flex-wrap:wrap; justify-content:center; margin-top:10px;'>", unsafe_allow_html=True)
        for label, value in stats.items():
            st.markdown(f"""
                <div style="min-width:150px; text-align:center; padding:15px; background-color:#f8f9fa; border-radius:10px;">
                <div style="font-size:24px; font-weight:bold; color:#0078D4;">{value}</div>
                <div style="font-size:14px; color:#666;">{label}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def display_json_response(result):
    """Display the raw JSON response."""
    with st.expander("View Raw JSON Response", expanded=False):
        st.json(result)

def show_use_cases():
    """Display use cases for Computer Vision in a horizontal layout."""
    st.subheader("🚀 Transformative Use Cases")
    
    # Define use cases
    use_cases = [
        {
            "title": "Retail Intelligence",
            "description": "Optimize store layouts, analyze customer behavior, and automate inventory management with visual AI.",
            "icon": "🛒"
        },
        {
            "title": "Content Moderation",
            "description": "Automatically filter inappropriate images and ensure brand safety across your digital platforms.",
            "icon": "🛡️"
        },
        {
            "title": "Document Processing",
            "description": "Extract text from forms, receipts, and documents to automate data entry and reduce manual work.",
            "icon": "📄"
        },
        {
            "title": "Quality Assurance",
            "description": "Detect defects, ensure product quality, and automate visual inspections in manufacturing.",
            "icon": "✓"
        },
        {
            "title": "Accessibility",
            "description": "Generate image descriptions for visually impaired users, making your content more accessible.",
            "icon": "♿"
        },
        {
            "title": "Smart Tagging",
            "description": "Automatically organize and categorize media libraries with AI-generated tags and metadata.",
            "icon": "🏷️"
        }
    ]
    
    # Create a 3-column layout for cards (2 rows of 3 cards)
    for i in range(0, len(use_cases), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(use_cases):
                case = use_cases[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div style="border: 1px solid #e6e6e6; border-radius: 10px; padding: 20px; height: 200px; margin-bottom: 15px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <div style="font-size: 36px; margin-bottom: 10px;">{case['icon']}</div>
                        <h3 style="color: #0078D4; font-size: 18px; margin-bottom: 10px;">{case['title']}</h3>
                        <p style="font-size: 14px; color: #555;">{case['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

def show_azure_information():
    """Display information about Azure Computer Vision."""
    with st.expander("About Azure Computer Vision", expanded=False):
        st.markdown("""
        ## Azure Computer Vision

        Azure Computer Vision is a cloud-based AI service that analyzes visual content in different ways, giving you information about what's in your images and videos.

        ### Key Capabilities

        - **Generate captions and descriptions** for images
        - **Detect and classify objects** in images with bounding boxes
        - **Recognize text (OCR)** from images of all types
        - **Detect people** and their locations in images
        - **Tag images** with relevant keywords based on thousands of recognizable objects
        - **Analyze content** for adult themes, violence, or other potentially sensitive content
        - **Extract rich information** from images to categorize and process visual data

        ### Benefits

        - **Accuracy**: State-of-the-art AI models trained on millions of diverse images
        - **Ease of integration**: Simple REST API calls or client library for various programming languages
        - **Scalability**: Process from a few images to millions of images
        - **Flexibility**: Use only the features you need for your specific use case
        - **Security**: Enterprise-grade security and compliance with Microsoft Azure
        """)

def show_code_example():
    """Display a code example for using Computer Vision."""
    with st.expander("Code Example", expanded=False):
        st.markdown("""
        ## Python Code Example

        Here's how to use Azure Computer Vision with Python:
        """)
        
        st.code("""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
endpoint = os.getenv('COMPUTER_VISION_ENDPOINT')
key = os.getenv('COMPUTER_VISION_KEY')

# Analyze an image
def analyze_image(image_path):
    # API endpoint for image analysis
    analyze_url = f"{endpoint}computervision/imageanalysis:analyze"
    
    # Request headers and parameters
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream"
    }
    
    params = {
        "features": "caption,tags,objects,people,read",
        "model-version": "latest",
        "language": "en",
        "api-version": "2023-10-01"
    }
    
    # Read image data
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    # Make request to Computer Vision API
    response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
    return response.json()

# Use the function
result = analyze_image("your_image.jpg")

# Print the caption
if 'captionResult' in result:
    print(f"Caption: {result['captionResult']['text']}")
    print(f"Confidence: {result['captionResult']['confidence']}")

# Print all detected objects
if 'objectsResult' in result and 'values' in result['objectsResult']:
    print("\nObjects detected:")
    for obj in result['objectsResult']['values']:
        print(f"- {obj['name']} (Confidence: {obj['confidence']:.2f})")
        """, language="python")

def show_pricing_info():
    """Display pricing information."""
    with st.expander("Pricing Information", expanded=False):
        st.markdown("""
        ## Azure Computer Vision Pricing

        Azure Computer Vision offers a flexible, pay-as-you-go pricing model:

        | Tier | Price per 1,000 Transactions |
        |------|------------------------------|
        | Free (F0) | First 20K transactions per month are free |
        | Standard (S1) | $1.00 - $5.00 depending on volume |

        ### Image Analysis Features
        
        Each call to analyze an image counts as a transaction. Multiple features in a single call (caption, tags, objects, etc.) count as a single transaction.

        ### Volume Discounts

        - 0-1M transactions: Standard pricing
        - 1M-5M transactions: 10% discount
        - 5M-10M transactions: 15% discount
        - 10M+ transactions: Contact Microsoft for custom pricing

        For the most current pricing information, visit the [Azure Computer Vision pricing page](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/computer-vision/).
        """)

def run_analysis(image_data, features, image=None):
    if not all([ENDPOINT, KEY, REGION]):
        st.error("""
        Missing Azure Computer Vision credentials. Please set the following environment variables in your .env file:
        - COMPUTER_VISION_ENDPOINT
        - COMPUTER_VISION_KEY
        - COMPUTER_VISION_REGION
        """)
        return
    
    # Run analysis
    result = analyze_image(image_data, features)
    
    if not result:
        st.error("Analysis failed. Please try again with a different image.")
        return
    
    # Display results
    st.subheader("Analysis Results")
    
    # First display overview
    display_analysis_overview(result)
    
    # Display results for selected features
    if "caption" in features:
        display_caption_results(result)
    
    if "denseCaptions" in features and image is not None:
        display_dense_captions(result, image)

    if "tags" in features:
        display_tags_results(result)
    
    if "objects" in features and image is not None:
        display_objects_results(result, image)
    
    if "read" in features and image is not None:
        display_text_results(result, image)
    
    # Display JSON response
    display_json_response(result)
    
    return result


def show_computer_vision():
    """Main function to display the Computer Vision demo."""
    set_custom_css()
    show_header()
    
    # Create main sections with tabs
    tab1, tab2 = st.tabs(["Try Computer Vision", "Learn More"])
    
    with tab1:
        st.subheader("Analyze your image with Azure Computer Vision")
        st.markdown("""
        Upload your own image or use one of our samples to discover what Azure Computer Vision can do!
        Select which analysis features you want to use and get instant results.
        """)
        
        # Get image
        image_data, image, source_type = show_image_upload_section()
        
        # Get analysis options
        if image_data:
            features = show_analysis_options()
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                # Run analysis
                run_analysis(image_data, features, image)
        else:
            st.warning("Please upload an image, provide an image URL, or select a sample image to begin.")
    
    with tab2:
        st.subheader("Azure Computer Vision Capabilities")
        st.markdown("""
        Azure Computer Vision is a cloud-based AI service that provides state-of-the-art computer vision capabilities.
        Analyze images to identify and extract visual data using the latest artificial intelligence algorithms.
        """)
        
        # Show use cases
        show_use_cases()
        
        # Show information sections
        col1, col2 = st.columns(2)
        
        with col1:
            show_azure_information()
            show_code_example()
        
        with col2:
            show_pricing_info()
            
            # Add a "Get Started" section
            st.markdown("""
            ## Get Started Today

            1. **Create an Azure account** if you don't have one already
            2. **Create a Computer Vision resource** in the Azure portal
            3. **Get your endpoint and API key** from the resource
            4. **Start integrating** Computer Vision into your applications

            Azure offers a free tier with 5,000 transactions per month, making it easy to get started without any upfront cost.
            """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Azure Computer Vision Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    show_computer_vision()