"""
Azure AI Search - Search and Relevance
-------------------------------------
This module demonstrates how to perform searches and tune relevance in Azure AI Search.
"""

import os
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizableTextQuery, 
    QueryType, 
    QueryCaptionType, 
    QueryAnswerType, 
    SemanticErrorMode
)
from openai import AzureOpenAI

def get_search_client(index_name="hr-documents-index"):
    """
    Get a search client for the specified index.
    
    Args:
        index_name (str): Name of the index
        
    Returns:
        SearchClient: The search client
    """
    # Load environment variables
    load_dotenv()
    
    # Get Azure Search credentials
    search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
    
    # Use API key authentication
    credential = AzureKeyCredential(search_admin_key)
    
    # Create SearchClient
    search_client = SearchClient(endpoint=search_endpoint, 
                                 index_name=index_name,
                                 credential=credential)
    
    return search_client

def get_openai_client():
    """
    Get an Azure OpenAI client.
    
    Returns:
        AzureOpenAI: The OpenAI client
    """
    # Load environment variables
    load_dotenv()
    
    # Get Azure OpenAI credentials
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    # Use DefaultAzureCredential for authentication
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    
    # Create OpenAI client
    openai_client = AzureOpenAI(
        api_version="2024-08-01-preview",
        azure_endpoint=openai_endpoint,
        azure_ad_token_provider=token_provider
    )
    
    return openai_client

def basic_keyword_search(search_client, query_text, top=5, filter_str=None):
    """
    Perform a basic keyword search.
    
    Args:
        search_client: The search client
        query_text (str): The search query
        top (int): Maximum number of results to return
        filter_str (str): Filter expression
        
    Returns:
        list: Search results
    """
    results = search_client.search(
        search_text=query_text,
        filter=filter_str,
        select=["chunk_id", "parent_id", "title", "content", "category", "document_type", "department", "chunk", "persons", "organizations", "locations", "keyPhrases"],
        highlight_fields="content,title,chunk",
        highlight_pre_tag="<b>",
        highlight_post_tag="</b>",
        top=top
    )
    
    return list(results)

def vector_search(search_client, query_text, top=5, filter_str=None):
    """
    Perform a vector search.
    
    Args:
        search_client: The search client
        query_text (str): The search query
        top (int): Maximum number of results to return
        filter_str (str): Filter expression
        
    Returns:
        list: Search results
    """
    vector_query = VectorizableTextQuery(
        text=query_text,
        k_nearest_neighbors=top,
        fields="content_vector"
    )
    
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        filter=filter_str,
        select=["chunk_id", "parent_id", "title", "content", "category", "document_type", "department", "chunk", "persons", "organizations", "locations", "keyPhrases"],
        top=top
    )
    
    return list(results)



def hybrid_search(search_client, query_text, top=5, filter_str=None):
    """
    Perform a hybrid search (combining keyword and vector search).
    
    Args:
        search_client: The search client
        query_text (str): The search query
        top (int): Maximum number of results to return
        filter_str (str): Filter expression
        
    Returns:
        list: Search results
    """
    vector_query = VectorizableTextQuery(
        text=query_text,
        k_nearest_neighbors=50,
        fields="content_vector"
    )
    
    results = search_client.search(
        search_text=query_text,
        vector_queries=[vector_query],
        filter=filter_str,
        select=["chunk_id", "parent_id", "title", "content", "category", "document_type", "department", "chunk", "persons", "organizations", "locations", "keyPhrases"],
        highlight_fields="content,title,chunk",
        highlight_pre_tag="<b>",
        highlight_post_tag="</b>",
        scoring_profile="hr-content-scoring",
        top=top
    )
    
    return list(results)

def semantic_hybrid_search(search_client, query_text, top=5, filter_str=None):
    """
    Perform a hybrid search with enhanced options for semantic search if available.
    Note: This is a compatibility version that works with older SDK versions.
    
    Args:
        search_client: The search client
        query_text (str): The search query
        top (int): Maximum number of results to return
        filter_str (str): Filter expression
        
    Returns:
        list: Search results
    """
    vector_query = VectorizableTextQuery(
        text=query_text,
        k_nearest_neighbors=50,
        fields="content_vector"
    )
    
    # Create the search parameters
    search_parameters = {
        "search_text": query_text,
        "vector_queries": [vector_query],
        "filter": filter_str,
        "select": ["chunk_id", "parent_id", "title", "content", "category", "document_type", "department", "chunk", "persons", "organizations", "locations", "keyPhrases"],
        "highlight_fields": "content,title,chunk",
        "highlight_pre_tag": "<b>",
        "highlight_post_tag": "</b>",
        "scoring_profile": "hr-content-scoring",
        "top": top
    }
    
    # Add semantic search options if available in SDK
    try:
        # Try to import necessary classes for semantic search
        from azure.search.documents.models import QueryType
        
        # If import successful, add semantic search parameters
        search_parameters["query_type"] = "semantic"
        search_parameters["semantic_configuration_name"] = "my-semantic-config"
    except ImportError:
        # If import fails, continue without semantic search
        print("Semantic search not available in this SDK version. Proceeding with basic search.")
        pass
    
    # Execute the search with the parameters
    results = search_client.search(**search_parameters)
    
    return list(results)


def generate_answer(query_text, search_results, model_deployment="gpt-4o-mini"):
    """
    Generate an answer using Azure OpenAI.
    
    Args:
        query_text (str): The search query
        search_results (list): Search results to use as context
        model_deployment (str): The OpenAI model deployment name
        
    Returns:
        str: The generated answer
    """
    # Get OpenAI client
    openai_client = get_openai_client()
    
    # Extract relevant information from search results
    formatted_results = []
    for i, result in enumerate(search_results):
        title = result.get('title', 'Untitled')
        source = f"Document: {title}"
        
        # Look for relevant content in the chunk field, which often contains the most focused text
        relevant_content = ""
        
        # First check if we have highlights in the chunk field
        if '@search.highlights' in result and 'chunk' in result['@search.highlights']:
            chunk_highlights = result['@search.highlights']['chunk']
            relevant_content = "\n".join([h.replace("<b>", "**").replace("</b>", "**") for h in chunk_highlights])
            source += f" (highlighted chunk)"
        
        # If no chunk highlights, check the full chunk
        elif 'chunk' in result and result['chunk']:
            relevant_content = result['chunk']
            source += f" (chunk)"
            
        # If no chunk, check content highlights
        elif '@search.highlights' in result and 'content' in result['@search.highlights']:
            content_highlights = result['@search.highlights']['content']
            relevant_content = "\n".join([h.replace("<b>", "**").replace("</b>", "**") for h in content_highlights])
            source += f" (highlighted content)"
            
        # Fallback to full content
        elif 'content' in result and result['content']:
            content = result['content']
            # Truncate if too long
            if len(content) > 1000:
                content = content[:1000] + "..."
            relevant_content = content
            source += f" (content)"
        
        if relevant_content:
            formatted_results.append(f"SOURCE {i+1}: {source}\n{relevant_content}")
    
    formatted_context = "\n\n" + "\n\n".join(formatted_results)
    
    # Define the prompt with explicit instructions to use the information in the context
    prompt = f"""
    You are an HR assistant that helps employees find information from the HR knowledge base.
    
    Answer the following query using ONLY the information provided in the search results below. 
    If the exact information is available in the search results, provide it clearly and directly.
    If related information is available but doesn't fully answer the question, provide what's available 
    and note what's missing.
    If no relevant information is present in the search results, say that you don't have enough information.
    
    PAY SPECIAL ATTENTION to information in the search results that might be related to the query,
    even if it doesn't exactly match the query keywords.
    
    Query: {query_text}
    
    Search Results:
    {formatted_context}
    """
    
    # Generate answer
    response = openai_client.chat.completions.create(
        model=model_deployment,
        messages=[
            {"role": "system", "content": "You are an HR assistant helping employees find information from company documents. Be precise and use information from the provided sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more focused answers
        max_tokens=500
    )
    
    return response.choices[0].message.content

def compare_search_methods(query_text, index_name="hr-documents-index"):
    """
    Compare different search methods for a given query.
    
    Args:
        query_text (str): The search query
        index_name (str): The search index name
        
    Returns:
        dict: Results from different search methods
    """
    # Get search client
    search_client = get_search_client(index_name)
    
    # Perform searches using different methods
    basic_results = basic_keyword_search(search_client, query_text)
    vector_results = vector_search(search_client, query_text)
    hybrid_results = hybrid_search(search_client, query_text)
    semantic_results = semantic_hybrid_search(search_client, query_text)
    
    # Calculate scores average for comparison
    basic_avg = sum(r['@search.score'] for r in basic_results) / len(basic_results) if basic_results else 0
    vector_avg = sum(r['@search.score'] for r in vector_results) / len(vector_results) if vector_results else 0
    hybrid_avg = sum(r['@search.score'] for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0
    semantic_avg = sum(r['@search.score'] for r in semantic_results) / len(semantic_results) if semantic_results else 0
    
    print(f"Average scores - Basic: {basic_avg:.2f}, Vector: {vector_avg:.2f}, Hybrid: {hybrid_avg:.2f}, Semantic: {semantic_avg:.2f}")
    
    # Print summary of each method's results
    print("\nBasic Keyword Search Results:")
    _print_method_results_summary(basic_results, "Basic")
    
    print("\nVector Search Results:")
    _print_method_results_summary(vector_results, "Vector")
    
    print("\nHybrid Search Results:")
    _print_method_results_summary(hybrid_results, "Hybrid")
    
    print("\nSemantic Hybrid Search Results:")
    _print_method_results_summary(semantic_results, "Semantic")
    
    # Return the results from each method
    return {
        "basic_keyword": basic_results,
        "vector": vector_results,
        "hybrid": hybrid_results,
        "semantic_hybrid": semantic_results
    }

def _print_method_results_summary(results, method_name):
    """Print a summary of search results for a specific method"""
    print(f"Method: {method_name} Search - Found {len(results)} results")
    
    for i, result in enumerate(results[:3]):  # Show top 3
        title = result.get('title', 'Untitled')
        score = result.get('@search.score', 0)
        print(f"{i+1}. {title} (Score: {score:.2f})")
        
        # Show highlights related to the query if available
        if '@search.highlights' in result and result['@search.highlights'] is not None:
            print("   Relevant highlights:")
            
            # Try to show chunk highlights first as they're usually more relevant
            if 'chunk' in result['@search.highlights']:
                for j, highlight in enumerate(result['@search.highlights']['chunk'][:2]):  # Show up to 2 highlights
                    clean_highlight = highlight.replace("<b>", "").replace("</b>", "")
                    if len(clean_highlight) > 100:
                        clean_highlight = clean_highlight[:100] + "..."
                    print(f"   - {clean_highlight}")
            # Fallback to content highlights
            elif 'content' in result['@search.highlights']:
                for j, highlight in enumerate(result['@search.highlights']['content'][:2]):  # Show up to 2 highlights
                    clean_highlight = highlight.replace("<b>", "").replace("</b>", "")
                    if len(clean_highlight) > 100:
                        clean_highlight = clean_highlight[:100] + "..."
                    print(f"   - {clean_highlight}")
        
        # For vector search (which doesn't provide highlights), show a content snippet
        elif method_name == "Vector" and 'chunk' in result:
            print("   Content snippet:")
            chunk = result.get('chunk', '')
            if len(chunk) > 100:
                chunk = chunk[:100] + "..."
            print(f"   - {chunk}")
            
        print()

def analyze_results(results):
    """
    Analyze and print details about search results.
    
    Args:
        results: Search results to analyze
    """
    print(f"Found {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result.get('title', 'No title')}")
        print(f"Score: {result.get('@search.score', 0):.2f}")
        
        # Print document metadata
        print(f"Category: {result.get('category', 'N/A')}")
        print(f"Document Type: {result.get('document_type', 'N/A')}")
        
        # Print highlighted snippets if available
        if '@search.highlights' in result:
            print("\nHighlights:")
            for field, highlights in result['@search.highlights'].items():
                for highlight in highlights:
                    print(f"  - {field}: {highlight}")
        
        # Print semantic captions if available (only if using semantic search)
        if '@search.captions' in result and result['@search.captions'] is not None:
            try:
                print("\nCaptions:")
                for caption in result['@search.captions']:
                    print(f"  - {caption.text} (Highlights: {caption.highlights})")
            except (TypeError, AttributeError) as e:
                print(f"\nUnable to print captions: {e}")
        
        print("-" * 80)


if __name__ == "__main__":
    # Example usage
    query = "What is the company policy on remote work?"
    
    print("\n=== Search and Answer ===")
    # Get search client
    search_client = get_search_client()
    
    # Perform semantic hybrid search
    results = semantic_hybrid_search(search_client, query)
    
    # Analyze the results
    analyze_results(results)
    
    # Generate an answer
    answer = generate_answer(query, results)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    
    # Compare different search methods
    print("\n=== Compare Search Methods ===")
    comparison = compare_search_methods(query)
    
    # Print result titles for each method
    for method, method_results in comparison.items():
        print(f"\n{method.replace('_', ' ').title()} Search:")
        for i, result in enumerate(method_results[:3]):  # Show top 3
            print(f"{i+1}. {result.get('title', 'No title')} (Score: {result.get('@search.score', 0):.2f})")