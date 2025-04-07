"""
Azure Document Intelligence Client module
Provides a centralized client connection for all Document Intelligence operations
"""

import os
from typing import Optional
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# Load environment variables
load_dotenv()

# Get credentials from environment variables
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
DOCUMENT_INTELLIGENCE_REGION = os.getenv("DOCUMENT_INTELLIGENCE_REGION")

def get_document_intelligence_client() -> Optional[DocumentIntelligenceClient]:
    """
    Initialize and return a Document Intelligence client
    
    Returns:
        DocumentIntelligenceClient or None: Initialized client or None if credentials not found
    """
    if not DOCUMENT_INTELLIGENCE_ENDPOINT or not DOCUMENT_INTELLIGENCE_KEY:
        print("Error: Document Intelligence credentials not found in environment variables!")
        return None
    
    try:
        credential = AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
        client = DocumentIntelligenceClient(
            endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT, 
            credential=credential
        )
        return client
    except Exception as e:
        print(f"Error initializing Document Intelligence client: {str(e)}")
        return None