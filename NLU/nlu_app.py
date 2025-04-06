import streamlit as st
import os
import json
import time
import random
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get CLU configuration
prediction_url = os.getenv('CLU_PREDICTION_ENDPOINT')
lang_key = os.getenv('LANGUAGE_SERVICE_KEY')
request_id = os.getenv('CLU_REQUEST_ID')
deployment = os.getenv('CLU_DEPLOYMENT_NAME')
project_name = os.getenv('CLU_PROJECT_NAME')

# Define intent colors for consistent visualization
INTENT_COLORS = {
    "GetRestaurantInfo": "#5D9CEC",  # Blue
    "RequestSpecialAccommodation": "#A0D468",  # Green
    "CheckAvailability": "#FFCE54",  # Yellow
    "BookTable": "#FC6E51",  # Orange
    "CancelReservation": "#ED5565",  # Red
    "ModifyReservation": "#AC92EC",  # Purple
    "None": "#AAB2BD"  # Gray
}

# Define entity colors for highlighting
ENTITY_COLORS = {
    "SpecialRequest": "#FC6E51",     # Orange
    "NumberOfPeople": "#5D9CEC",     # Blue
    "ReservationTime": "#A0D468",    # Green
    "CustomerName": "#AC92EC",       # Purple
    "PhoneNumber": "#FFCE54",        # Yellow
    "ReservationID": "#ED5565"       # Red
}

# ----- CLU Functions -----

def analyze_text(query_text, language="en"):
    """
    Analyze text using the CLU model.
    
    Args:
        query_text (str): Text to analyze
        language (str): Language code (default: "en")
    
    Returns:
        dict: CLU analysis results
    """
    # Request headers
    headers = {
        'Ocp-Apim-Subscription-Key': lang_key,
        'Apim-Request-Id': request_id,
        'Content-Type': 'application/json'
    }
    
    # Request body
    data = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "id": "1",
                "text": query_text,
                "modality": "text",
                "language": language,
                "participantId": "1"
            }
        },
        "parameters": {
            "projectName": project_name,
            "verbose": True,
            "deploymentName": deployment,
            "stringIndexType": "TextElement_V8"
        }
    }
    
    # Make the request
    try:
        response = requests.post(prediction_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling CLU service: {str(e)}")
        return None

def extract_results(results):
    """Extract useful information from CLU results."""
    if not results:
        return None
    
    prediction = results.get('result', {}).get('prediction', {})
    
    # Get all intents with confidence scores
    intents = {}
    for intent in prediction.get('intents', []):
        intents[intent['category']] = intent['confidenceScore']
    
    # Get top intent
    top_intent = prediction.get('topIntent')
    
    # Get entities
    entities = []
    for entity in prediction.get('entities', []):
        entities.append({
            'category': entity['category'],
            'text': entity['text'],
            'confidence': entity['confidenceScore']
        })
    
    return {
        'query': results['result']['query'],
        'intents': intents,
        'top_intent': top_intent,
        'entities': entities
    }

# ----- Visualization Functions -----

def create_intent_confidence_chart(intents):
    """Create a bar chart for intent confidence scores."""
    if not intents:
        return None
    
    # Prepare data
    intent_names = list(intents.keys())
    confidence_scores = list(intents.values())
    
    # Assign colors based on predefined mapping
    colors = [INTENT_COLORS.get(intent, "#AAB2BD") for intent in intent_names]
    
    # Create figure
    fig = px.bar(
        x=confidence_scores, 
        y=intent_names,
        orientation='h',
        title="Intent Recognition Confidence Scores",
        labels={"x": "Confidence Score", "y": "Intent"},
        color=intent_names,
        color_discrete_map={intent: INTENT_COLORS.get(intent, "#AAB2BD") for intent in intent_names}
    )
    
    # Update layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis=dict(range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def get_highlighted_text(text, entities):
    """Create HTML with highlighted entities in the text."""
    if not entities:
        return text
    
    # Sort entities by their position in text to handle overlapping entities
    # Use a more sophisticated approach - create character-level tags
    char_tags = [{"start": False, "end": False, "entity": None} for _ in range(len(text) + 1)]
    
    for entity in entities:
        # Find all occurrences of the entity text
        entity_text = entity['text']
        start_idx = 0
        while True:
            start_idx = text.find(entity_text, start_idx)
            if start_idx == -1:
                break
                
            end_idx = start_idx + len(entity_text)
            
            # Mark start and end of entity
            if not char_tags[start_idx]["start"]:  # Don't overwrite existing tags
                char_tags[start_idx] = {
                    "start": True, 
                    "end": False, 
                    "entity": entity['category']
                }
            
            if not char_tags[end_idx]["end"]:
                char_tags[end_idx] = {
                    "start": False, 
                    "end": True, 
                    "entity": entity['category']
                }
                
            start_idx = end_idx
    
    # Build the HTML string with proper open/close tags
    html_text = ""
    open_entities = []
    
    for i, char in enumerate(text):
        # Check if we need to close any tags
        if i < len(text) and char_tags[i]["end"]:
            if open_entities:  # Close the most recently opened entity
                html_text += "</span>"
                open_entities.pop()
        
        # Check if we need to open any tags
        if char_tags[i]["start"]:
            entity_category = char_tags[i]["entity"]
            color = ENTITY_COLORS.get(entity_category, "#AAB2BD")
            html_text += f'<span style="background-color: {color}40; padding: 2px; border-radius: 3px; border-bottom: 2px solid {color};" title="{entity_category}">'
            open_entities.append(entity_category)
        
        # Add the character
        html_text += char
    
    # Close any remaining open tags at the end
    while open_entities:
        html_text += "</span>"
        open_entities.pop()
    
    return html_text

def create_entity_legend():
    """Create a legend for entity highlighting."""
    # Create a cleaner display for entity types
    cols = st.columns(3)
    i = 0
    
    for entity, color in ENTITY_COLORS.items():
        with cols[i % 3]:
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                f'<div style="background-color: {color}; width: 15px; height: 15px; margin-right: 5px; border-radius: 3px;"></div>'
                f'<span>{entity}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        i += 1

def create_entity_table(entities):
    """Create a formatted table of recognized entities using Streamlit's native components."""
    if not entities:
        return st.info("No entities detected.")
    
    # Create a DataFrame for display
    entity_data = []
    for entity in entities:
        entity_data.append({
            "Entity Type": entity['category'],
            "Value": entity['text'],
            "Confidence": f"{entity['confidence']*100:.1f}%"
        })
    
    if entity_data:
        df = pd.DataFrame(entity_data)
        st.dataframe(df, use_container_width=True)

def create_radar_chart(intents):
    """Create a radar chart for the top 5 intent confidence scores."""
    if not intents:
        return None
    
    # Get top 5 intents by confidence
    top_intents = dict(sorted(intents.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Prepare data
    categories = list(top_intents.keys())
    values = list(top_intents.values())
    
    # Close the loop for radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(93, 156, 236, 0.5)',
        line=dict(color='#5D9CEC', width=2),
        name='Intent Confidence'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Intent Recognition Pattern',
        height=350,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig

def create_entity_extraction_sankey(entities):
    """Create a Sankey diagram visualizing entity extraction flow."""
    if not entities:
        return None
    
    # Not enough data for a real Sankey, so we'll simulate one
    # to show the conceptual flow from query -> entity categories -> extracted values
    
    # Prepare data
    labels = ["Query"]  # Start with "Query" node
    
    # Add entity category nodes
    category_indices = {}
    for idx, entity in enumerate(entities):
        category = entity['category']
        if category not in category_indices:
            category_indices[category] = len(labels)
            labels.append(category)
    
    # Add entity value nodes
    value_indices = {}
    for entity in entities:
        value = entity['text']
        if value not in value_indices:
            value_indices[value] = len(labels)
            labels.append(value)
    
    # Create source-target pairs
    sources = []
    targets = []
    values = []
    
    # Query -> Entity Category
    for category, idx in category_indices.items():
        sources.append(0)  # Query is index 0
        targets.append(idx)
        # Count instances of this category
        category_count = sum(1 for e in entities if e['category'] == category)
        values.append(category_count)
    
    # Entity Category -> Entity Value
    for entity in entities:
        category_idx = category_indices[entity['category']]
        value_idx = value_indices[entity['text']]
        sources.append(category_idx)
        targets.append(value_idx)
        values.append(1)  # Each entity instance has a value of 1
    
    # Create figure with fixed color format that works with Plotly
    node_colors = ["rgba(93, 156, 236, 0.8)"]  # Start with blue for query node
    
    # Add colors for entity categories
    for label in labels[1:len(category_indices)+1]:
        color = ENTITY_COLORS.get(label, "#AAB2BD")
        # Convert hex to rgba
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        node_colors.append(f"rgba({r}, {g}, {b}, 0.8)")
    
    # Add colors for entity values (neutral gray)
    node_colors.extend(["rgba(170, 178, 189, 0.6)"] * len(value_indices))
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(170, 178, 189, 0.3)"  # Use a single valid color with transparency
        )
    )])
    
    fig.update_layout(
        title_text="Entity Extraction Flow",
        font_size=12,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_confidence_distribution(intents, entities):
    """Create a histogram showing confidence score distribution."""
    if not intents:
        return None
    
    # Combine confidence scores from intents and entities
    confidence_scores = list(intents.values())
    
    if entities:
        confidence_scores.extend([entity['confidence'] for entity in entities])
    
    # Create histogram
    fig = px.histogram(
        x=confidence_scores, 
        nbins=10,
        range_x=[0, 1],
        labels={"x": "Confidence Score", "y": "Count"},
        title="Confidence Score Distribution",
        color_discrete_sequence=["#5D9CEC"]
    )
    
    # Add mean line
    mean_confidence = np.mean(confidence_scores)
    fig.add_vline(x=mean_confidence, line_dash="dash", line_color="#ED5565",
                 annotation_text=f"Mean: {mean_confidence:.2f}",
                 annotation_position="top right")
    
    fig.update_layout(
        bargap=0.1,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_confidence_gauge(confidence, title="Overall Confidence"):
    """Create a gauge chart for confidence visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "#5D9CEC"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 50], "color": "#FFCDD2"},
                {"range": [50, 75], "color": "#FFE0B2"},
                {"range": [75, 100], "color": "#C8E6C9"}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

# ----- Session State Management -----

def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
        
    if 'last_user_message' not in st.session_state:
        st.session_state.last_user_message = ""
        
    if 'assistant_responded' not in st.session_state:
        st.session_state.assistant_responded = False

# ----- UI Components -----

def show_nlu_demo():
    """Main function to display the NLU demo interface."""
    st.title("Conversational Language Understanding (CLU)")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    with st.sidebar:
        st.image("https://azure.microsoft.com/svghandler/cognitive-services/?width=600&height=315", width=200)
        st.subheader("About CLU")
        st.markdown("""
        **Conversational Language Understanding (CLU)** is part of Azure AI Language that helps to:
        
        - 🔍 **Understand user intent** from natural language input
        - 🏷️ **Extract entities** (key pieces of information)
        - 🧠 **Build intelligent applications** that understand human language
        """)
        
        st.subheader("Features")
        st.markdown("""
        - ⚡ **Real-time processing** of natural language
        - 🌍 **Multi-language support**
        - 📊 **Confidence scoring** for intents and entities
        - 🔄 **Continuous improvement** through model training
        """)
        
        st.subheader("Our Model")
        st.markdown(f"""
        - **Project**: {project_name}
        - **Deployment**: {deployment}
        """)
        
        # Add a reset button
        if st.button("Reset Conversation"):
            st.session_state.conversation_history = []
            st.session_state.analysis_history = []
            st.session_state.current_analysis = None
            st.session_state.last_user_message = ""
            st.session_state.assistant_responded = False
            st.rerun()
    
    # Create a two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # If we have a current analysis, show visual explainer
        if st.session_state.current_analysis:
            st.subheader("Understanding Your Request")
            
            analysis = st.session_state.current_analysis
            query = analysis['query']
            
            # Display highlighted text with entity recognition
            st.markdown("### What we heard")
            highlighted_text = get_highlighted_text(query, analysis['entities'])
            st.markdown(f'<div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; font-size: 16px; line-height: 1.5;">{highlighted_text}</div>', unsafe_allow_html=True)
            
            # Display entity legend if entities were detected
            if analysis['entities']:
                st.markdown("### Entities Recognized")
                entity_legend = create_entity_legend()
                st.markdown(entity_legend, unsafe_allow_html=True)
                
                # Display entity table
                entity_table = create_entity_table(analysis['entities'])
                st.markdown(entity_table, unsafe_allow_html=True)
            
            # Display intent recognition
            st.markdown("### Intent Recognition")
            
            # Create columns for top intent and confidence
            intent_col1, intent_col2 = st.columns([1, 1])
            with intent_col1:
                top_intent = analysis['top_intent']
                intent_color = INTENT_COLORS.get(top_intent, "#AAB2BD")
                st.markdown(f'<div style="border-left: 5px solid {intent_color}; padding-left: 10px;"><h4>Detected Intent</h4><p style="font-size: 18px; font-weight: bold;">{top_intent}</p></div>', unsafe_allow_html=True)
            
            with intent_col2:
                # Show confidence gauge for top intent
                top_confidence = analysis['intents'].get(top_intent, 0)
                fig = create_confidence_gauge(top_confidence, "Intent Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            # Show intent confidence chart
            intent_chart = create_intent_confidence_chart(analysis['intents'])
            if intent_chart:
                st.plotly_chart(intent_chart, use_container_width=True)
            
            # Advanced visualizations in tabs
            st.markdown("### Advanced Analysis")
            tab1, tab2, tab3 = st.tabs(["Intent Pattern", "Entity Flow", "Confidence Analysis"])
            
            with tab1:
                radar_chart = create_radar_chart(analysis['intents'])
                if radar_chart:
                    st.plotly_chart(radar_chart, use_container_width=True)
                    st.markdown("""
                    This radar chart shows the confidence pattern across intents, helping visualize how the model "thinks" about the input.
                    """)
            
            with tab2:
                if analysis['entities']:
                    sankey = create_entity_extraction_sankey(analysis['entities'])
                    if sankey:
                        st.plotly_chart(sankey, use_container_width=True)
                        st.markdown("""
                        This flow diagram visualizes how entities are extracted from the query, showing the relationship between entity types and values.
                        """)
                else:
                    st.info("No entities were detected in this query.")
            
            with tab3:
                # Confidence distribution
                confidence_dist = create_confidence_distribution(analysis['intents'], analysis['entities'])
                if confidence_dist:
                    st.plotly_chart(confidence_dist, use_container_width=True)
                    st.markdown("""
                    This histogram shows the distribution of confidence scores across all intents and entities, 
                    providing insight into how certain the model is about its predictions.
                    """)
    
    with col2:
        # Create a chat-like interface
        st.subheader("Try It Yourself")
        
        # Sample queries with buttons
        st.markdown("### Sample Queries")
        
        sample_queries = [
            "Do you have outdoor seating?",
            "I need a table for 4 people tomorrow at 7 PM",
            "Can I get a quiet table away from the kitchen?",
            "What time do you close on weekends?",
            "Is your restaurant wheelchair accessible?",
            "I want to change my reservation from Tuesday to Wednesday"
        ]
        
        # 2x3 grid of sample query buttons
        for i in range(0, len(sample_queries), 2):
            col_a, col_b = st.columns(2)
            
            with col_a:
                if i < len(sample_queries) and st.button(sample_queries[i], key=f"sample_{i}"):
                    # Clear previous conversation if starting new
                    if sample_queries[i] not in [msg["content"] for msg in st.session_state.conversation_history if msg["role"] == "user"]:
                        st.session_state.conversation_history.append({"role": "user", "content": sample_queries[i]})
                        
                        # Trigger analysis
                        with st.spinner("Analyzing..."):
                            results = analyze_text(sample_queries[i])
                            analysis = extract_results(results)
                            
                            if analysis:
                                st.session_state.current_analysis = analysis
                                st.session_state.analysis_history.append(analysis)
                                st.rerun()
            
            with col_b:
                if i+1 < len(sample_queries) and st.button(sample_queries[i+1], key=f"sample_{i+1}"):
                    # Clear previous conversation if starting new
                    if sample_queries[i+1] not in [msg["content"] for msg in st.session_state.conversation_history if msg["role"] == "user"]:
                        st.session_state.conversation_history.append({"role": "user", "content": sample_queries[i+1]})
                        
                        # Trigger analysis
                        with st.spinner("Analyzing..."):
                            results = analyze_text(sample_queries[i+1])
                            analysis = extract_results(results)
                            
                            if analysis:
                                st.session_state.current_analysis = analysis
                                st.session_state.analysis_history.append(analysis)
                                st.rerun()
        
        # Custom query input
        with st.form(key="query_form"):
            query_input = st.text_input("Enter your own query:", placeholder="Type a restaurant-related query...")
            submit_button = st.form_submit_button("Analyze")
            
            if submit_button and query_input:
                # Check if this query is already in conversation
                if query_input not in [msg["content"] for msg in st.session_state.conversation_history if msg["role"] == "user"]:
                    st.session_state.conversation_history.append({"role": "user", "content": query_input})
                    
                    # Trigger analysis
                    with st.spinner("Analyzing..."):
                        results = analyze_text(query_input)
                        analysis = extract_results(results)
                        
                        if analysis:
                            st.session_state.current_analysis = analysis
                            st.session_state.analysis_history.append(analysis)
                            st.rerun()
        
        # Display conversation history
        st.markdown("### Conversation History")
        
        # Draw a chat-like conversation
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <div style="background-color: #5D9CEC30; border-radius: 10px 10px 0 10px; padding: 10px; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                    <div style="background-color: #F5F7FA; border-radius: 10px 10px 10px 0; padding: 10px; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Generate system response based on analysis
        if (st.session_state.current_analysis and 
            len(st.session_state.conversation_history) > 0 and
            st.session_state.conversation_history[-1]["role"] == "user"):
            
            # Get last user message
            last_user_message = st.session_state.conversation_history[-1]["content"]
            
            # Check if we need to respond
            if (last_user_message != st.session_state.last_user_message or
                not st.session_state.assistant_responded):
                
                # Create a response based on the detected intent and entities
                analysis = st.session_state.current_analysis
            analysis = st.session_state.current_analysis
            
            response = ""
            
            if analysis['top_intent'] == "GetRestaurantInfo":
                response = "I'd be happy to provide information about our restaurant. "
                
                # Check if any specific info was requested
                info_requested = False
                for entity in analysis['entities']:
                    if entity['category'] == "SpecialRequest" and "vegan" in entity['text'].lower():
                        response += "Yes, we offer a variety of vegan options on our menu."
                        info_requested = True
                    elif entity['category'] == "SpecialRequest" and "outdoor" in entity['text'].lower():
                        response += "Yes, we have a beautiful outdoor patio available for seating."
                        info_requested = True
                
                if not info_requested:
                    response += "We're open Monday-Thursday 11am-10pm, Friday-Saturday 11am-11pm, and Sunday 10am-9pm. Our specialties include fresh seafood and house-made pasta."
            
            elif analysis['top_intent'] == "CheckAvailability":
                response = "Let me check availability for you. "
                
                # Extract time and party size if available
                time_entity = next((e for e in analysis['entities'] if e['category'] == "ReservationTime"), None)
                people_entity = next((e for e in analysis['entities'] if e['category'] == "NumberOfPeople"), None)
                
                if time_entity and people_entity:
                    response += f"We do have availability for {people_entity['text']} at {time_entity['text']}. Would you like to book a table?"
                elif time_entity:
                    response += f"We have several table sizes available at {time_entity['text']}. How many people will be in your party?"
                elif people_entity:
                    response += f"We have several time slots available for a party of {people_entity['text']}. What time would you prefer?"
                else:
                    response += "What date and time are you looking for, and how many people will be in your party?"
            
            elif analysis['top_intent'] == "BookTable":
                response = "I'd be happy to book a table for you. "
                
                # Extract details if available
                time_entity = next((e for e in analysis['entities'] if e['category'] == "ReservationTime"), None)
                people_entity = next((e for e in analysis['entities'] if e['category'] == "NumberOfPeople"), None)
                name_entity = next((e for e in analysis['entities'] if e['category'] == "CustomerName"), None)
                
                if time_entity and people_entity:
                    response += f"I've reserved a table for {people_entity['text']} at {time_entity['text']}. "
                    if name_entity:
                        response += f"The reservation is under the name {name_entity['text']}. "
                    else:
                        response += "May I have your name for the reservation?"
                else:
                    missing_info = []
                    if not time_entity:
                        missing_info.append("preferred time")
                    if not people_entity:
                        missing_info.append("number of guests")
                    
                    response += f"I'll need your {' and '.join(missing_info)} to complete the booking."
            
            elif analysis['top_intent'] == "RequestSpecialAccommodation":
                response = "We'll be happy to accommodate your special request. "
                
                # Check specific accommodation
                for entity in analysis['entities']:
                    if entity['category'] == "SpecialRequest":
                        if "high chair" in entity['text'].lower() or "child" in entity['text'].lower():
                            response += "We'll have a high chair ready for your child."
                        elif "quiet" in entity['text'].lower() or "away from kitchen" in entity['text'].lower():
                            response += "I've noted your preference for a quiet table away from the kitchen."
                        elif "wheelchair" in entity['text'].lower() or "accessible" in entity['text'].lower():
                            response += "Yes, our restaurant is fully wheelchair accessible with ramp access and accessible restrooms."
                        else:
                            response += f"I've added a note for your request: '{entity['text']}'."
                
                if not any(e['category'] == "SpecialRequest" for e in analysis['entities']):
                    response += "What specific accommodation would you like us to prepare for your visit?"
            
            elif analysis['top_intent'] == "ModifyReservation":
                response = "I'd be happy to modify your existing reservation. "
                
                # Check for reservation ID
                res_id = next((e for e in analysis['entities'] if e['category'] == "ReservationID"), None)
                
                if res_id:
                    response += f"I've located your reservation #{res_id['text']}. What changes would you like to make?"
                else:
                    response += "Could you provide your name or reservation number so I can locate your booking?"
            
            elif analysis['top_intent'] == "CancelReservation":
                response = "I can help you cancel your reservation. "
                
                # Check for reservation ID or name
                res_id = next((e for e in analysis['entities'] if e['category'] == "ReservationID"), None)
                name = next((e for e in analysis['entities'] if e['category'] == "CustomerName"), None)
                
                if res_id:
                    response += f"I've cancelled reservation #{res_id['text']}. You'll receive a confirmation email shortly."
                elif name:
                    response += f"I've found a reservation under the name {name['text']}. Would you like me to cancel it?"
                else:
                    response += "Could you provide your name or reservation number so I can locate your booking?"
            
            else:  # Handle "None" intent or anything else
                response = "I'm not sure I understood your request. Could you please rephrase? I can help with restaurant information, checking availability, making reservations, or special accommodations."
            
            # Simulate typing for a more realistic feel
            with st.spinner("Assistant is typing..."):
                time.sleep(1.5)  # Simulate response time
            
            # Add the response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Add a bottom section with additional explanations
    with st.expander("About This Demo", expanded=False):
        st.markdown("""
        ### How it Works
        
        This demo showcases Azure's Conversational Language Understanding (CLU) service, which:
        
        1. **Analyzes natural language** text to determine user intent
        2. **Extracts key information** (entities) from user queries
        3. **Provides confidence scores** for intent recognition and entity extraction
        
        The model has been trained on restaurant reservation scenarios and can understand various intents such as:
        
        - Getting restaurant information
        - Checking availability
        - Booking tables
        - Requesting special accommodations
        - Modifying or canceling reservations
        
        ### Business Value
        
        Implementing CLU can help businesses:
        
        - **Automate customer interactions** with intelligent virtual assistants
        - **Streamline workflows** by automatically extracting key information from text
        - **Understand customer needs** more accurately
        - **Reduce operational costs** while improving customer experience
        - **Gain insights** from customer conversations
        
        This technology can be applied to various industries beyond restaurants, including healthcare, finance, retail, and more.
        """)

# Main function to be called from streamlit_app.py
def show_nlu():
    # Verify configuration
    if not all([prediction_url, lang_key, request_id, deployment, project_name]):
        missing = [var for var, val in {
            'CLU_PREDICTION_ENDPOINT': prediction_url,
            'LANGUAGE_SERVICE_KEY': lang_key,
            'CLU_REQUEST_ID': request_id,
            'CLU_DEPLOYMENT_NAME': deployment,
            'CLU_PROJECT_NAME': project_name
        }.items() if not val]
        
        st.error(f"Error: Missing environment variables: {', '.join(missing)}")
        st.info("Please ensure these are set in your .env file")
        return
    
    # Show the main demo
    show_nlu_demo()

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Azure NLU Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Run the app
    show_nlu()