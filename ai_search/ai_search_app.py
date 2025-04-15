"""
Azure AI Search - Streamlit Application
--------------------------------------
This module provides a Streamlit interface for demonstrating Azure AI Search capabilities
with HR documents.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import AzureOpenAI
import time
from datetime import datetime

# Function to highlight search terms in text
def highlight_text(text, highlights=None, max_length=300):
    if not highlights:
        # If no highlights provided, truncate text and return
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    # If highlights are provided, create a formatted string with highlights
    display_text = text
    for highlight in highlights:
        # Replace the highlighted text with a formatted version
        highlight_clean = highlight.replace("<b>", "").replace("</b>", "")
        if highlight_clean in display_text:
            display_text = display_text.replace(
                highlight_clean, 
                highlight.replace("<b>", "<span style='background-color: #FFFF00; font-weight: bold;'>").replace("</b>", "</span>")
            )
    
    # Truncate the text if it's too long
    if len(display_text) > max_length:
        return display_text[:max_length] + "..."
    
    return display_text

# Get Azure clients
def get_search_client():
    """Initialize and return the Azure AI Search client"""
    # Load environment variables
    load_dotenv()
    
    # Get Azure Search credentials
    search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
    index_name = "hr-documents-index"
    
    # Use API key authentication
    try:
        credential = AzureKeyCredential(search_admin_key)
        
        # Create and return SearchClient
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential
        )
        return search_client
    except Exception as e:
        st.error(f"Error initializing Azure AI Search client: {str(e)}")
        return None

def get_openai_client():
    """Initialize and return the Azure OpenAI client"""
    # Load environment variables
    load_dotenv()
    
    # Get Azure OpenAI credentials
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    try:
        # Create and return OpenAI client with API key
        openai_client = AzureOpenAI(
            api_version="2024-08-01-preview",
            azure_endpoint=openai_endpoint,
            api_key=openai_api_key
        )
        return openai_client
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI client: {str(e)}")
        return None

# Search functions
def execute_search(search_client, query, search_type="semantic_hybrid", filters=None, top_n=5):
    """
    Execute a search with the specified parameters
    
    Args:
        search_client: The search client
        query: The search query
        search_type: Type of search to perform
        filters: Filters to apply
        top_n: Number of results to retrieve
        
    Returns:
        Search results
    """
    if not search_client:
        st.error("Search client is not initialized")
        return []
    
    try:
        # Select fields to retrieve
        select_fields = ["chunk_id", "parent_id", "title", "content", "category", "document_type", 
                         "department", "chunk", "persons", "organizations", "locations", "keyPhrases"]
                         
        if search_type == "keyword":
            # Basic keyword search
            results = search_client.search(
                search_text=query,
                filter=filters,
                select=select_fields,
                highlight_fields="content,title,chunk",
                highlight_pre_tag="<b>",
                highlight_post_tag="</b>",
                top=top_n
            )
        elif search_type == "vector":
            # Vector search
            vector_query = VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=top_n,
                fields="content_vector"
            )
            
            results = search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filters,
                select=select_fields,
                top=top_n
            )
        elif search_type == "hybrid":
            # Hybrid search
            vector_query = VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=50,
                fields="content_vector"
            )
            
            results = search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=filters,
                select=select_fields,
                highlight_fields="content,title,chunk",
                highlight_pre_tag="<b>",
                highlight_post_tag="</b>",
                scoring_profile="hr-content-scoring",
                top=top_n
            )
        else:  # semantic_hybrid (default)
            # Semantic hybrid search - with compatibility for older SDK versions
            vector_query = VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=50,
                fields="content_vector"
            )
            
            # Create the search parameters
            search_parameters = {
                "search_text": query,
                "vector_queries": [vector_query],
                "filter": filters,
                "select": select_fields,
                "highlight_fields": "content,title,chunk",
                "highlight_pre_tag": "<b>",
                "highlight_post_tag": "</b>",
                "scoring_profile": "hr-content-scoring",
                "top": top_n
            }
            
            # Add semantic search options if available in the SDK
            try:
                from azure.search.documents.models import QueryType, SemanticSearchOptions, QueryCaptionType, QueryAnswerType
                
                # Add semantic search parameters
                search_parameters["query_type"] = QueryType.SEMANTIC
                search_parameters["semantic_search_options"] = SemanticSearchOptions(
                    configuration_name="my-semantic-config",
                    query_caption=QueryCaptionType.EXTRACTIVE,
                    query_answer=QueryAnswerType.EXTRACTIVE
                )
            except (ImportError, AttributeError):
                # If import fails, try a simpler approach
                try:
                    search_parameters["query_type"] = "semantic"
                    search_parameters["semantic_configuration_name"] = "my-semantic-config"
                except:
                    # If all semantic options fail, continue with standard hybrid search
                    pass
            
            # Execute the search with the parameters
            results = search_client.search(**search_parameters)
        
        # Normalize all results to ensure consistent structure
        normalized_results = []
        for result in results:
            try:
                # Convert to dictionary if needed
                if not isinstance(result, dict):
                    result_dict = dict(result)
                else:
                    result_dict = result.copy()
                
                # Add all required fields with safe defaults to prevent 'NoneType' errors
                for field in ["title", "content", "chunk", "document_type", "department"]:
                    if field not in result_dict:
                        result_dict[field] = ""
                
                # Make sure these are all lists for safe iteration
                for field in ["persons", "organizations", "locations", "keyPhrases"]:
                    if field not in result_dict or result_dict[field] is None:
                        result_dict[field] = []
                    elif not isinstance(result_dict[field], list):
                        result_dict[field] = [result_dict[field]]
                
                # Ensure @search.highlights exists (for vector search)
                if '@search.highlights' not in result_dict:
                    result_dict['@search.highlights'] = {}
                    
                # Ensure it's not None
                if result_dict['@search.highlights'] is None:
                    result_dict['@search.highlights'] = {}
                
                # Always make score available
                if '@search.score' not in result_dict:
                    result_dict['@search.score'] = 0.0
                
                normalized_results.append(result_dict)
            except Exception as e:
                st.error(f"Error normalizing result: {str(e)}")
                # Add a simple dict with required fields
                normalized_results.append({
                    "title": "Result processing error",
                    "content": f"Error processing result: {str(e)}",
                    "chunk": "",
                    "document_type": "unknown",
                    "department": "unknown",
                    "persons": [],
                    "organizations": [],
                    "locations": [],
                    "keyPhrases": [],
                    "@search.highlights": {},
                    "@search.score": 0.0
                })
                
        return normalized_results
    except Exception as e:
        st.error(f"Error performing search: {str(e)}")
        return []

def generate_answer(openai_client, query, search_results, model_name="gpt-4o-mini"):
    """Generate an answer using Azure OpenAI based on search results"""
    if not openai_client or not search_results:
        return "Unable to generate an answer due to missing information or configuration."
    
    try:
        # Format search results as context
        context_items = []
        
        for i, result in enumerate(search_results[:3]):  # Use top 3 results
            title = result.get('title', 'Untitled')
            source = f"Document: {title}"
            
            # Look for relevant content to use in the context
            relevant_text = ""
            
            # First check for highlights in the chunk field - often the most relevant
            if '@search.highlights' in result and 'chunk' in result['@search.highlights']:
                relevant_text = "\n".join([h.replace("<b>", "**").replace("</b>", "**") 
                                          for h in result['@search.highlights']['chunk']])
                source += " (highlighted chunk)"
            
            # If no chunk highlights, check the full chunk
            elif 'chunk' in result and result['chunk']:
                relevant_text = result['chunk']
                source += " (full chunk)"
                
            # Next try content highlights
            elif '@search.highlights' in result and 'content' in result['@search.highlights']:
                relevant_text = "\n".join([h.replace("<b>", "**").replace("</b>", "**") 
                                          for h in result['@search.highlights']['content']])
                source += " (highlighted content)"
                
            # If still no relevant text, try semantic captions
            elif '@search.captions' in result and result['@search.captions']:
                try:
                    captions_text = [caption.text for caption in result['@search.captions']]
                    relevant_text = "\n".join(captions_text)
                    source += " (semantic captions)"
                except (AttributeError, TypeError):
                    # Fall back to content if captions can't be processed
                    pass
                    
            # Final fallback - use content field
            if not relevant_text and 'content' in result:
                content = result.get('content', '')
                if len(content) > 800:
                    content = content[:800] + "..."
                relevant_text = content
                source += " (content excerpt)"
            
            # Add extracted entities and key phrases for additional context
            entities_info = ""
            if result.get('persons') or result.get('organizations'):
                entities_info = "\nEntities mentioned: "
                if result.get('persons'):
                    entities_info += f"People: {', '.join(result['persons'][:3])}. "
                if result.get('organizations'):
                    entities_info += f"Organizations: {', '.join(result['organizations'][:3])}. "
            
            key_phrases = ""
            if result.get('keyPhrases'):
                key_phrases = f"\nKey concepts: {', '.join(result['keyPhrases'][:5])}"
            
            if relevant_text:
                context_items.append(f"SOURCE {i+1}: {source}\n{relevant_text}{entities_info}{key_phrases}")
        
        context = "\n\n" + "\n\n".join(context_items)
        
        # Create a more directive prompt that encourages finding relevant information
        prompt = f"""
        You are an HR assistant that helps employees find information from the HR knowledge base.
        
        Answer the following query using ONLY the information provided in the context below.
        If the exact information is available in the context, provide it clearly and directly.
        If related information is available but doesn't fully answer the question, provide what's available and note what's missing.
        If no relevant information is present in the context, say that you don't have enough information.
        
        PAY SPECIAL ATTENTION to information in the context that might be related to the query,
        even if it doesn't exactly match the query keywords. Look for policy information that relates to
        the query topic.
        
        QUERY: {query}
        
        CONTEXT: {context}
        """
        
        # Generate answer with lower temperature for more focused responses
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an HR assistant helping employees find information from company documents. Be precise and use information from the provided sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "I encountered an error while trying to generate an answer. Please try again later."

# Streamlit UI
def show_ai_search():
    
    # Create session state for storing search results with safe defaults
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    
    if "answer" not in st.session_state:
        st.session_state.answer = ""
    
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
        
    if "search_time" not in st.session_state:
        st.session_state.search_time = 0.0
        
    if "error_message" not in st.session_state:
        st.session_state.error_message = ""
    
    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://azure.microsoft.com/svghandler/search/?width=600&height=315", width=80)
    with col2:
        st.title("HR Knowledge Base Explorer")
        st.markdown("""
        <div style="margin-bottom: 20px;">
        <span style="font-size: 1.2em; color: #666;">Find HR policies, procedures, and guidelines using advanced AI search capabilities.</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for filters and options
    with st.sidebar:
        st.header("Search Options")
        
        # Search method selection
        st.subheader("Search Technology")
        search_type = st.radio(
            "Select search method:",
            ["Semantic Hybrid", "Hybrid", "Vector", "Keyword"],
            index=0,
            help="Semantic Hybrid: Combines keywords, vectors, and semantic understanding\n"
                 "Hybrid: Combines keywords and vectors\n"
                 "Vector: Uses embeddings for semantic similarity\n"
                 "Keyword: Basic text matching"
        )
        
        # Map selection to API parameter
        search_type_map = {
            "Semantic Hybrid": "semantic_hybrid",
            "Hybrid": "hybrid",
            "Vector": "vector",
            "Keyword": "keyword"
        }
        
        # Store the selected search type in session state for reference in display logic
        st.session_state.selected_search_type = search_type_map[search_type]
        
        # Filters section
        st.subheader("Filter Results")
        
        # Get document types and departments from actual data
        document_types = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        selected_doc_types = st.multiselect("Document Type", document_types)
        
        # Use file paths as departments
        departments = ["Employee Handbook", "Policies", "Procedures", "Benefits", "HR Forms"]
        selected_departments = st.multiselect("Document Category", departments)
        
        # Build filter string
        filter_clauses = []
        if selected_doc_types:
            clause = " or ".join([f"document_type eq '{doc}'" for doc in selected_doc_types])
            filter_clauses.append(f"({clause})")
        
        if selected_departments:
            # We're now dealing with Edm.String for department, not a collection
            clause = " or ".join([f"search.in(department, '{dept}')" for dept in selected_departments])
            filter_clauses.append(f"({clause})")
        
        filter_string = " and ".join(filter_clauses) if filter_clauses else None
        
        # Result settings
        st.subheader("Results")
        top_n = st.slider("Number of results", min_value=3, max_value=20, value=8)
        
        # Additional options
        st.subheader("Additional Options")
        generate_ai_answer = st.checkbox("Generate AI Answer", value=True, 
                                       help="Use Azure OpenAI to generate an answer based on search results")
        
        # Search history
        if st.session_state.search_history:
            st.subheader("Recent Searches")
            for i, past_query in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"{past_query[:30]}...", key=f"history_{i}"):
                    st.session_state.last_query = past_query
                    st.rerun()
    
    # Main content area
    # Create a prominent search bar
    with st.container():
        st.markdown("""
        <style>
        .search-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        <div class="search-container">
        </div>
        """, unsafe_allow_html=True)
        
        query = st.text_input(
            "Ask a question about HR policies, procedures, or guidelines:",
            value=st.session_state.last_query,
            placeholder="E.g., What is our remote work policy?",
            key="search_query"
        )
        
        search_col1, search_col2 = st.columns([5, 1])
        with search_col2:
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        # Display example questions
        with st.expander("Need inspiration? Try these example questions"):
            example_col1, example_col2, example_col3 = st.columns(3)
            
            with example_col1:
                if st.button("What is our parental leave policy?"):
                    st.session_state.last_query = "What is our parental leave policy?"
                    st.rerun()
                if st.button("How do I request time off?"):
                    st.session_state.last_query = "How do I request time off?"
                    st.rerun()
            
            with example_col2:
                if st.button("What is the policy on remote work?"):
                    st.session_state.last_query = "What is the policy on remote work?"
                    st.rerun()
                if st.button("How do I submit an expense report?"):
                    st.session_state.last_query = "How do I submit an expense report?"
                    st.rerun()
            
            with example_col3:
                if st.button("What are the steps for onboarding a new employee?"):
                    st.session_state.last_query = "What are the steps for onboarding a new employee?"
                    st.rerun()
                if st.button("What is the dress code policy?"):
                    st.session_state.last_query = "What is the dress code policy?"
                    st.rerun()
    
    # Execute search when requested
    if (query and (search_button or query != st.session_state.last_query)):
        # Clear any previous error
        st.session_state.error_message = ""
        
        # Update search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
        
        st.session_state.last_query = query
        
        # Initialize clients
        search_client = get_search_client()
        if generate_ai_answer:
            openai_client = get_openai_client()
        else:
            openai_client = None
        
        # Execute search
        with st.spinner("Searching knowledge base..."):
            try:
                start_time = time.time()
                results = execute_search(
                    search_client, 
                    query, 
                    search_type=search_type_map[search_type],
                    filters=filter_string,
                    top_n=top_n
                )
                search_time = time.time() - start_time
                
                # Make sure results is a list, even if empty
                if results is None:
                    results = []
                    
                st.session_state.search_results = results
                st.session_state.search_time = search_time  # Save search time in session state
            except Exception as e:
                st.session_state.error_message = f"Error performing search: {str(e)}"
                st.error(st.session_state.error_message)
                st.session_state.search_results = []
        
        # Generate answer if requested
        if generate_ai_answer and openai_client and st.session_state.search_results and not st.session_state.error_message:
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(openai_client, query, st.session_state.search_results)
                    st.session_state.answer = answer
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.answer = "I encountered an error while trying to generate an answer. Please try again later."
                    st.session_state.error_message = error_msg
    
    # Display search results if available
    if st.session_state.search_results:
        # Show search summary
        st.markdown(f"### Search Results")
        st.markdown(f"Found {len(st.session_state.search_results)} results for: **{st.session_state.last_query}**")
        
        # Display AI answer if available
        if st.session_state.answer:
            st.markdown("### AI Assistant Response")
            st.markdown(
                f"""
                <div style="background-color: #f0f7fb; padding: 20px; border-radius: 10px; 
                border-left: 5px solid #2196F3; margin-bottom: 20px;">
                {st.session_state.answer}
                <div style="font-size: 0.8em; color: #666; margin-top: 10px;">
                <em>Note: This answer was generated based on search results. Always verify information in official HR documents.</em>
                </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Document Results", "Analytics"])
        
        with tab1:
            # Display search results in an interactive way
            for i, result in enumerate(st.session_state.search_results):
                # Create an expandable card for each result
                with st.expander(f"{i+1}. {result.get('title', 'Untitled Document')}", expanded=i==0):
                    result_col1, result_col2 = st.columns([3, 1])
                    
                    with result_col1:
                        try:
                            # Check if we have a chunk (excerpt) with highlights
                            highlights = result.get('@search.highlights', {})
                            
                            if highlights and highlights.get('chunk'):
                                st.markdown("**Relevant Excerpt:**")
                                for highlight in highlights.get('chunk', []):
                                    formatted_highlight = highlight.replace(
                                        "<b>", "<span style='background-color: #FFFF00; font-weight: bold;'>"
                                    ).replace("</b>", "</span>")
                                    st.markdown(f"<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 10px;'>{formatted_highlight}</div>", unsafe_allow_html=True)
                            
                            # If we have highlights from content, show those
                            elif highlights and highlights.get('content'):
                                st.markdown("**Relevant Content:**")
                                for highlight in highlights.get('content', []):
                                    formatted_highlight = highlight.replace(
                                        "<b>", "<span style='background-color: #FFFF00; font-weight: bold;'>"
                                    ).replace("</b>", "</span>")
                                    st.markdown(f"<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 10px;'>{formatted_highlight}</div>", unsafe_allow_html=True)
                            
                            # Otherwise, show a chunk preview
                            elif result.get('chunk'):
                                content = result.get('chunk', 'No content available')
                                if content and len(content) > 800:
                                    content = content[:800] + "..."
                                st.markdown("**Content Preview:**")
                                st.markdown(f"<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
                                
                            # Fallback to content preview
                            else:
                                content = result.get('content', 'No content available')
                                if content and len(content) > 800:
                                    content = content[:800] + "..."
                                st.markdown("**Content Preview:**")
                                st.markdown(f"<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
                            
                            # Add a visualization to display extracted entities and key phrases
                            has_entities = False
                            
                            # Check if any entities or key phrases exist
                            if (result.get('persons') or result.get('organizations') or 
                                result.get('locations') or result.get('keyPhrases')):
                                
                                has_entities = True
                                st.markdown("**Extracted Information:**")
                                entity_col1, entity_col2 = st.columns(2)
                                
                                with entity_col1:
                                    if result.get('persons') or result.get('organizations') or result.get('locations'):
                                        st.markdown("**Named Entities:**")
                                        
                                        persons = result.get('persons', [])
                                        if persons:
                                            if not isinstance(persons, list):
                                                persons = [persons]
                                            st.markdown(f"- **People**: {', '.join(persons)}")
                                            
                                        organizations = result.get('organizations', [])
                                        if organizations:
                                            if not isinstance(organizations, list):
                                                organizations = [organizations]
                                            st.markdown(f"- **Organizations**: {', '.join(organizations)}")
                                            
                                        locations = result.get('locations', [])
                                        if locations:
                                            if not isinstance(locations, list):
                                                locations = [locations]
                                            st.markdown(f"- **Locations**: {', '.join(locations)}")
                                
                                with entity_col2:
                                    key_phrases = result.get('keyPhrases', [])
                                    if key_phrases:
                                        if not isinstance(key_phrases, list):
                                            key_phrases = [key_phrases]
                                            
                                        st.markdown("**Key Phrases:**")
                                        phrases = key_phrases[:7]  # Limit to 7 key phrases
                                        for phrase in phrases:
                                            st.markdown(f"- {phrase}")
                                        if len(key_phrases) > 7:
                                            st.markdown(f"*...and {len(key_phrases) - 7} more*")
                                            
                            # If no entities or key phrases, add some space
                            if not has_entities:
                                st.write("")  # Add some space
                        except Exception as e:
                            st.error(f"Error displaying result content: {str(e)}")
                            st.markdown("**Content Preview:**")
                            st.markdown("Error displaying content. Please try a different search type or query.")
                    
                    with result_col2:
                        # Document metadata and score
                        st.markdown("**Document Information**")
                        
                        # Display document type with an icon
                        doc_type = result.get('document_type', 'Unknown')
                        doc_type_icon = {
                            "application/pdf": "📋",
                            "text/plain": "📝",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "📄"
                        }.get(doc_type, "📄")
                        
                        st.markdown(f"**Type:** {doc_type_icon} {doc_type}")
                        
                        # Display department (source path)
                        dept = result.get('department', 'Unknown')
                        if isinstance(dept, str) and '/' in dept:
                            # Try to extract a more readable name from the path
                            parts = dept.split('/')
                            if len(parts) > 1:
                                dept_name = parts[-1].replace('%20', ' ')
                            else:
                                dept_name = dept
                        else:
                            dept_name = dept
                            
                        st.markdown(f"**Source:** {dept_name}")
                        
                        # Display last updated date if available
                        if 'last_updated' in result:
                            try:
                                # Format the date
                                date_obj = result['last_updated']
                                if isinstance(date_obj, str):
                                    date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                                formatted_date = date_obj.strftime("%b %d, %Y")
                                st.markdown(f"**Last Updated:** {formatted_date}")
                            except:
                                st.markdown(f"**Last Updated:** {result['last_updated']}")
                        
                        # Display relevance score with a gauge
                        score = result.get('@search.score', 0)
                        st.markdown(f"**Relevance Score:** {score:.2f}")
                        
                        # Create a score gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=min(score, 10),  # Cap at 10 for visualization
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 10], 'tickwidth': 1},
                                'bar': {'color': "royalblue"},
                                'steps': [
                                    {'range': [0, 3.33], 'color': "lightgray"},
                                    {'range': [3.33, 6.66], 'color': "gray"},
                                    {'range': [6.66, 10], 'color': "darkblue"}
                                ],
                            },
                            title={'text': ""}
                        ))
                        
                        fig.update_layout(
                            height=150,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        
                        # Each gauge must have a truly unique key
                        gauge_key = f"score_gauge_{i}_{result.get('chunk_id', i)}"
                        st.plotly_chart(fig, use_container_width=True, key=gauge_key)
                        
                        # Add a "View Full Document" button (in a real application, this would link to the document)
                        st.button("View Full Document", key=f"view_doc_{i}", use_container_width=True)
        
        with tab2:
            # Analytics and visualizations of search results
            if st.session_state.search_results:
                # Extract data for visualization
                result_data = []
                for r in st.session_state.search_results:
                    doc_type = r.get('document_type', 'Unknown')
                    dept = r.get('department', 'Unknown')
                    
                    # Clean up department name for display
                    if isinstance(dept, str) and '/' in dept:
                        parts = dept.split('/')
                        if len(parts) > 1:
                            dept_name = parts[-1].replace('%20', ' ')
                        else:
                            dept_name = dept
                    else:
                        dept_name = dept
                    
                    result_data.append({
                        "title": r.get('title', 'Untitled'),
                        "type": doc_type,
                        "department": dept_name,
                        "score": r.get('@search.score', 0)
                    })
                
                df = pd.DataFrame(result_data)
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Document distribution by type
                    type_counts = df['type'].value_counts().reset_index()
                    type_counts.columns = ['Document Type', 'Count']
                    
                    fig = px.pie(
                        type_counts,
                        values='Count',
                        names='Document Type',
                        title='Results by Document Type',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    
                    # Use a unique key based on search type and timestamp to avoid conflicts
                    pie_key = f"doc_type_pie_{st.session_state.selected_search_type}_{int(time.time())}"
                    st.plotly_chart(fig, use_container_width=True, key=pie_key)
                
                with col2:
                    # Document distribution by department
                    dept_counts = df['department'].value_counts().reset_index()
                    dept_counts.columns = ['Department', 'Count']
                    
                    fig = px.bar(
                        dept_counts,
                        x='Department',
                        y='Count',
                        title='Results by Source',
                        color='Department',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Use a unique key based on search type and timestamp to avoid conflicts
                    bar_key = f"dept_bar_{st.session_state.selected_search_type}_{int(time.time())}"
                    st.plotly_chart(fig, use_container_width=True, key=bar_key)
                
                # Relevance score distribution
                st.subheader(f"Relevance Scores ({search_type} Search)")
                
                # Sort by score
                df_sorted = df.sort_values('score', ascending=False)
                df_sorted = df_sorted.drop_duplicates('title')  # Remove duplicates from multiple departments
                
                fig = px.bar(
                    df_sorted,
                    x='title',
                    y='score',
                    color='score',
                    color_continuous_scale='Viridis',
                    labels={'title': 'Document', 'score': 'Relevance Score'},
                    title='Document Relevance Ranking'
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    xaxis_title="",
                    yaxis_title="Relevance Score"
                )
                
                # Use a unique key based on search type and timestamp to avoid conflicts
                relevance_key = f"relevance_bar_{st.session_state.selected_search_type}_{int(time.time())}"
                st.plotly_chart(fig, use_container_width=True, key=relevance_key)
                
                # Display search method comparison
                st.info(f"""
                **About {search_type} Search:**
                
                {'📚 **Semantic Hybrid Search** combines traditional keyword matching with vector similarity and semantic understanding to deliver the most accurate and contextually relevant results.' if search_type == 'Semantic Hybrid' else ''}
                {'🔄 **Hybrid Search** combines keyword matching with vector similarity to find both exact matches and semantically similar content.' if search_type == 'Hybrid' else ''}
                {'🧠 **Vector Search** uses embeddings to find documents that are semantically similar to your query, even if they dont contain the exact words.' if search_type == 'Vector' else ''}
                {'🔍 **Keyword Search** uses traditional text matching to find documents containing the words in your query.' if search_type == 'Keyword' else ''}
                
                This search was completed in {st.session_state.get("search_time", 0.0):.2f} seconds.
                """)
    else:
        # Display welcome message and feature highlights when no search has been performed
        st.markdown("""
        ### Welcome to the HR Knowledge Base Explorer!
        
        This tool helps you find HR policies, procedures, and guidelines quickly using advanced AI search technology.
        Enter your query above to get started.
        """)
        
        # Feature highlights
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
            #### 🔍 Intelligent Search
            Our advanced search technology understands what you're looking for, even if you don't use the exact words found in documents.
            """)
        
        with feature_col2:
            st.markdown("""
            #### 🤖 AI-Powered Answers
            Get direct answers to your questions, generated from relevant HR documents in the knowledge base.
            """)
        
        with feature_col3:
            st.markdown("""
            #### 📊 Visual Analytics
            See the distribution of search results across document types and departments to better understand your results.
            """)
        
        # Add a decorative element
        st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <img src="https://azure.microsoft.com/svghandler/search/?width=600&height=315" width="120px">
            <p style="color: #666; font-style: italic; margin-top: 20px;">Powered by Azure AI Search and Azure OpenAI</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_ai_search()