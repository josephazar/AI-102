#!/usr/bin/env python
"""
Azure Conversational Language Understanding (CLU) Test Script
------------------------------------------------------------
This script tests your deployed CLU project for restaurant reservations.
"""

import os
import json
import requests
from dotenv import load_dotenv
import pprint

# Load environment variables
load_dotenv()

# Get CLU configuration
prediction_url = os.getenv('CLU_PREDICTION_ENDPOINT')
lang_key = os.getenv('LANGUAGE_SERVICE_KEY')
request_id = os.getenv('CLU_REQUEST_ID')
deployment = os.getenv('CLU_DEPLOYMENT_NAME')
project_name = os.getenv('CLU_PROJECT_NAME')

# Verify configuration
if not all([prediction_url, lang_key, request_id, deployment, project_name]):
    missing = [var for var, val in {
        'CLU_PREDICTION_ENDPOINT': prediction_url,
        'LANGUAGE_SERVICE_KEY': lang_key,
        'CLU_REQUEST_ID': request_id,
        'CLU_DEPLOYMENT_NAME': deployment,
        'CLU_PROJECT_NAME': project_name
    }.items() if not val]
    
    print(f"Error: Missing environment variables: {', '.join(missing)}")
    print("Please ensure these are set in your .env file")
    exit(1)

# Headers for Azure Language Service
headers = {
    'Ocp-Apim-Subscription-Key': lang_key,
    'Apim-Request-Id': request_id,
    'Content-Type': 'application/json'
}

def analyze_text(query_text, language="en"):
    """
    Analyze text using the CLU model.
    
    Args:
        query_text (str): Text to analyze
        language (str): Language code (default: "en")
    
    Returns:
        dict: CLU analysis results
    """
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
        print(f"Error calling CLU service: {e}")
        return None

def format_results(results):
    """Format and display the CLU results nicely."""
    if not results:
        print("No results to display")
        return
    
    prediction = results.get('result', {}).get('prediction', {})
    
    # Get top intent and score
    top_intent = prediction.get('topIntent')
    top_score = next((i['confidenceScore'] for i in prediction.get('intents', []) 
                    if i['category'] == top_intent), 0)
    
    # Get entities
    entities = prediction.get('entities', [])
    
    # Print results
    print("\n" + "="*50)
    print(f"QUERY: {results['result']['query']}")
    print("="*50)
    print(f"\nTOP INTENT: {top_intent} (confidence: {top_score:.4f})")
    
    # Display all intents and scores
    print("\nALL INTENTS:")
    for intent in prediction.get('intents', []):
        print(f"  - {intent['category']}: {intent['confidenceScore']:.4f}")
    
    # Display entities
    if entities:
        print("\nRECOGNIZED ENTITIES:")
        for entity in entities:
            print(f"  - {entity['category']}: '{entity['text']}' (confidence: {entity['confidenceScore']:.4f})")
    else:
        print("\nNo entities recognized")
    
    print("\n" + "="*50)

def main():
    """Main function to test the CLU model with sample queries."""
    print("\nAzure Conversational Language Understanding (CLU) Test Script")
    print("-----------------------------------------------------------")
    print(f"Project: {project_name}")
    print(f"Deployment: {deployment}")
    
    # Sample queries to test
    test_queries = [
        "Do you offer vegan options?",
        "What are your business hours?",
        "I'd like a quiet table away from the kitchen",
        "We'll need a high chair for our child",
        "Are there any tables for a party of 8 tomorrow night?",
        "Check availability for 4 people on Saturday evening"
    ]
    
    print("\nTesting with sample queries...\n")
    
    # Test with sample queries
    for query in test_queries:
        results = analyze_text(query)
        format_results(results)
    
    # Interactive mode
    print("\nInteractive mode (type 'exit' to quit):")
    while True:
        query = input("\nEnter a query: ")
        if query.lower() == 'exit':
            break
        
        results = analyze_text(query)
        format_results(results)

if __name__ == "__main__":
    main()