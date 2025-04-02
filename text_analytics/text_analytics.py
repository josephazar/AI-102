"""
Azure Text Analytics - Core Functions
------------------------------------
This module provides simple, reusable functions for Azure Text Analytics service.
Use this as a reference implementation for Azure AI-102 exam preparation.
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

def initialize_text_analytics_client():
    """
    Initialize the Azure Text Analytics client using environment variables.
    
    Returns:
        TextAnalyticsClient: Initialized client or None if credentials are missing
    """
    # Load environment variables
    load_dotenv()
    
    # Get Azure configuration
    endpoint = os.getenv('COG_SERVICE_ENDPOINT')
    key = os.getenv('COG_SERVICE_KEY')
    
    if not endpoint or not key:
        print("Error: Azure credentials not found! Please ensure your .env file contains COG_SERVICE_ENDPOINT and COG_SERVICE_KEY values.")
        return None
    
    try:
        # Create Azure Text Analytics client
        credential = AzureKeyCredential(key)
        client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
        return client
    except Exception as e:
        print(f"Error initializing Azure client: {str(e)}")
        return None

def process_in_batches(client, documents, operation_func, batch_size=5):
    """
    Process documents in batches to comply with Azure Text Analytics service limits.
    
    Args:
        client: The Text Analytics client
        documents: List of documents to process
        operation_func: Function to call on each batch (e.g., client.analyze_sentiment)
        batch_size: Maximum number of documents per batch
        
    Returns:
        List of combined results
    """
    if not client or not documents:
        return None
    
    all_results = []
    
    try:
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = operation_func(documents=batch)
            all_results.extend(batch_results)
        
        return all_results
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return None

def analyze_sentiment(client, texts, languages=None):
    """
    Analyze sentiment of one or more texts.
    
    Args:
        client (TextAnalyticsClient): The Text Analytics client
        texts (list or str): A single text string or a list of text strings to analyze
        languages (list or str, optional): Language code(s) corresponding to the texts
        
    Returns:
        list: Results of sentiment analysis or None if error occurs
    """
    if not client:
        print("Client not initialized")
        return None
    
    try:
        # Handle single text vs list of texts
        if isinstance(texts, str):
            texts = [texts]
            
        if languages and isinstance(languages, str):
            languages = [languages]
        
        # Prepare documents
        documents = []
        for i, text in enumerate(texts):
            doc = {"id": str(i), "text": text}
            if languages and i < len(languages):
                doc["language"] = languages[i]
            documents.append(doc)
        
        # Call Azure service in batches
        return process_in_batches(client, documents, client.analyze_sentiment)
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return None

def extract_key_phrases(client, texts, languages=None):
    """
    Extract key phrases from one or more texts.
    
    Args:
        client (TextAnalyticsClient): The Text Analytics client
        texts (list or str): A single text string or a list of text strings
        languages (list or str, optional): Language code(s) corresponding to the texts
        
    Returns:
        list: Results of key phrase extraction or None if error occurs
    """
    if not client:
        print("Client not initialized")
        return None
    
    try:
        # Handle single text vs list of texts
        if isinstance(texts, str):
            texts = [texts]
            
        if languages and isinstance(languages, str):
            languages = [languages]
        
        # Prepare documents
        documents = []
        for i, text in enumerate(texts):
            doc = {"id": str(i), "text": text}
            if languages and i < len(languages):
                doc["language"] = languages[i]
            documents.append(doc)
        
        # Call Azure service in batches
        return process_in_batches(client, documents, client.extract_key_phrases)
    except Exception as e:
        print(f"Error extracting key phrases: {str(e)}")
        return None

def detect_language(client, texts):
    """
    Detect the language of one or more texts.
    
    Args:
        client (TextAnalyticsClient): The Text Analytics client
        texts (list or str): A single text string or a list of text strings
        
    Returns:
        list: Results of language detection or None if error occurs
    """
    if not client:
        print("Client not initialized")
        return None
    
    try:
        # Handle single text vs list of texts
        if isinstance(texts, str):
            texts = [texts]
        
        # Prepare documents
        documents = [{"id": str(i), "text": text} for i, text in enumerate(texts)]
        
        # Call Azure service in batches
        return process_in_batches(client, documents, client.detect_language)
    except Exception as e:
        print(f"Error detecting language: {str(e)}")
        return None

def recognize_entities(client, texts, languages=None):
    """
    Recognize entities in one or more texts.
    
    Args:
        client (TextAnalyticsClient): The Text Analytics client
        texts (list or str): A single text string or a list of text strings
        languages (list or str, optional): Language code(s) corresponding to the texts
        
    Returns:
        list: Results of entity recognition or None if error occurs
    """
    if not client:
        print("Client not initialized")
        return None
    
    try:
        # Handle single text vs list of texts
        if isinstance(texts, str):
            texts = [texts]
            
        if languages and isinstance(languages, str):
            languages = [languages]
        
        # Prepare documents
        documents = []
        for i, text in enumerate(texts):
            doc = {"id": str(i), "text": text}
            if languages and i < len(languages):
                doc["language"] = languages[i]
            documents.append(doc)
        
        # Call Azure service in batches
        return process_in_batches(client, documents, client.recognize_entities)
    except Exception as e:
        print(f"Error recognizing entities: {str(e)}")
        return None

def recognize_linked_entities(client, texts, languages=None):
    """
    Recognize linked entities in one or more texts.
    
    Args:
        client (TextAnalyticsClient): The Text Analytics client
        texts (list or str): A single text string or a list of text strings
        languages (list or str, optional): Language code(s) corresponding to the texts
        
    Returns:
        list: Results of linked entity recognition or None if error occurs
    """
    if not client:
        print("Client not initialized")
        return None
    
    try:
        # Handle single text vs list of texts
        if isinstance(texts, str):
            texts = [texts]
            
        if languages and isinstance(languages, str):
            languages = [languages]
        
        # Prepare documents
        documents = []
        for i, text in enumerate(texts):
            doc = {"id": str(i), "text": text}
            if languages and i < len(languages):
                doc["language"] = languages[i]
            documents.append(doc)
        
        # Call Azure service in batches
        return process_in_batches(client, documents, client.recognize_linked_entities)
    except Exception as e:
        print(f"Error recognizing linked entities: {str(e)}")
        return None

# Example function showing how to format results for easy viewing
def print_sentiment_analysis_results(results):
    """
    Print sentiment analysis results in a readable format.
    
    Args:
        results: Response from the analyze_sentiment function
    """
    if not results:
        print("No results to display")
        return
    
    for doc in results:
        if doc.is_error:
            print(f"Document {doc.id} had an error: {doc.error}")
            continue
            
        print(f"Document {doc.id} sentiment: {doc.sentiment}")
        print(f"Overall scores: Positive={doc.confidence_scores.positive:.2f}, "
              f"Neutral={doc.confidence_scores.neutral:.2f}, "
              f"Negative={doc.confidence_scores.negative:.2f}")
        
        print("Sentence sentiment:")
        for i, sentence in enumerate(doc.sentences):
            print(f"  Sentence {i+1} sentiment: {sentence.sentiment}")
            print(f"  Scores: Positive={sentence.confidence_scores.positive:.2f}, "
                  f"Neutral={sentence.confidence_scores.neutral:.2f}, "
                  f"Negative={sentence.confidence_scores.negative:.2f}")
            print(f"  Text: {sentence.text}")
            
        print("---" * 10)

def print_key_phrases_results(results):
    """
    Print key phrase extraction results in a readable format.
    
    Args:
        results: Response from the extract_key_phrases function
    """
    if not results:
        print("No results to display")
        return
    
    for doc in results:
        if doc.is_error:
            print(f"Document {doc.id} had an error: {doc.error}")
            continue
            
        print(f"Document {doc.id} key phrases:")
        for phrase in doc.key_phrases:
            print(f"  - {phrase}")
            
        print("---" * 10)

def print_language_detection_results(results):
    """
    Print language detection results in a readable format.
    
    Args:
        results: Response from the detect_language function
    """
    if not results:
        print("No results to display")
        return
    
    for doc in results:
        if doc.is_error:
            print(f"Document {doc.id} had an error: {doc.error}")
            continue
            
        print(f"Document {doc.id} language: {doc.primary_language.name}")
        print(f"ISO6391 name: {doc.primary_language.iso6391_name}")
        print(f"Confidence score: {doc.primary_language.confidence_score:.4f}")
            
        print("---" * 10)

def print_entity_recognition_results(results):
    """
    Print entity recognition results in a readable format.
    
    Args:
        results: Response from the recognize_entities function
    """
    if not results:
        print("No results to display")
        return
    
    for doc in results:
        if doc.is_error:
            print(f"Document {doc.id} had an error: {doc.error}")
            continue
            
        print(f"Document {doc.id} entities:")
        for entity in doc.entities:
            print(f"  - Text: {entity.text}")
            print(f"    Category: {entity.category}")
            if entity.subcategory:
                print(f"    Subcategory: {entity.subcategory}")
            print(f"    Confidence score: {entity.confidence_score:.4f}")
            
        print("---" * 10)

def print_linked_entity_recognition_results(results):
    """
    Print linked entity recognition results in a readable format.
    
    Args:
        results: Response from the recognize_linked_entities function
    """
    if not results:
        print("No results to display")
        return
    
    for doc in results:
        if doc.is_error:
            print(f"Document {doc.id} had an error: {doc.error}")
            continue
            
        print(f"Document {doc.id} linked entities:")
        for entity in doc.entities:
            print(f"  - Name: {entity.name}")
            print(f"    ID: {entity.data_source}")
            print(f"    URL: {entity.url}")
            print(f"    Data source: {entity.data_source}")
            
            print("    Matches:")
            for match in entity.matches:
                print(f"      - Text: {match.text}")
                print(f"        Confidence score: {match.confidence_score:.4f}")
            
        print("---" * 10)

# Example usage
def main():
    """
    Example of how to use the functions in this module.
    """
    # Initialize client
    client = initialize_text_analytics_client()
    if not client:
        return
    
    # Example texts
    texts = [
        "I had a wonderful trip to Seattle last week. The weather was perfect and the Space Needle was amazing!",
        "The food at the restaurant was terrible and the service was even worse.",
        "Azure Cognitive Services provide powerful AI capabilities that are easy to integrate into applications."
    ]
    
    print("\n=== Sentiment Analysis ===")
    results = analyze_sentiment(client, texts)
    print_sentiment_analysis_results(results)
    
    print("\n=== Key Phrase Extraction ===")
    results = extract_key_phrases(client, texts)
    print_key_phrases_results(results)
    
    print("\n=== Language Detection ===")
    texts_multilingual = [
        "Hello world!",
        "Hola mundo!",
        "Bonjour tout le monde!"
    ]
    results = detect_language(client, texts_multilingual)
    print_language_detection_results(results)
    
    print("\n=== Entity Recognition ===")
    results = recognize_entities(client, texts)
    print_entity_recognition_results(results)
    
    print("\n=== Linked Entity Recognition ===")
    results = recognize_linked_entities(client, texts)
    print_linked_entity_recognition_results(results)

if __name__ == "__main__":
    main()