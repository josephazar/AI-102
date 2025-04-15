"""
Azure AI Search - Index Creation
-------------------------------
This module demonstrates how to create an Azure AI Search index
with vector search capabilities for HR documents.
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    ScoringProfile,
    TextWeights
)

def create_hr_documents_index(index_name="hr-documents-index"):
    """
    Create an Azure AI Search index for HR documents with vector search capabilities.
    
    Args:
        index_name (str): Name of the index to create
        
    Returns:
        SearchIndex: The created or updated index
    """
    # Load environment variables
    load_dotenv()
    
    # Get Azure Search credentials
    search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
    
    # Use API key authentication
    credential = AzureKeyCredential(search_admin_key)
    
    # Create SearchIndexClient
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    
    # Delete the index if it already exists
    try:
        index_client.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")
    except Exception as e:
        print(f"Index '{index_name}' does not exist or could not be deleted: {e}")

    # Define fields for the index
    fields = [
        SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                facetable=True,
                analyzer_name="keyword"  # Explicitly set the keyword analyzer
            ),
            SearchField(
                name="parent_id",
                type=SearchFieldDataType.String,
                filterable=True,
                analyzer_name="keyword"  # Good practice for parent key field
            ),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True, searchable=True),
        SearchField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True, searchable=True),
        SearchField(name="last_updated", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        SearchField(name="department", type=SearchFieldDataType.String, filterable=True, facetable=True, searchable=True),
        SearchField(name="chunk", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                    vector_search_dimensions=1536, vector_search_profile_name="my-vector-profile"),
        SearchField(name="persons", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=True),
        SearchField(name="organizations", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=True),
        SearchField(name="locations", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=True),
        SearchField(name="keyPhrases", type=SearchFieldDataType.Collection(SearchFieldDataType.String), searchable=True, filterable=True),
    ]
    
    # Configure vector search with scalar quantization for optimization
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw-config",
                vectorizer_name="my-vectorizer",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="my-vectorizer",
                kind="azureOpenAI",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=openai_endpoint,
                    deployment_name=embedding_model,
                    model_name="text-embedding-ada-002"
                ),
            ),
        ],
    )
    
    # Configure semantic search for better relevance
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            keywords_fields=[
                SemanticField(field_name="department"), 
                SemanticField(field_name="category"),
                SemanticField(field_name="document_type")
            ],
            content_fields=[
                SemanticField(field_name="content"), 
                SemanticField(field_name="chunk")
            ]
        )
    )
    
    semantic_search = SemanticSearch(configurations=[semantic_config])
    
    # Define scoring profiles for improved relevance
    scoring_profiles = [
        ScoringProfile(
            name="hr-content-scoring",
            text_weights=TextWeights(
                weights={
                    "title": 5.0,
                    "content": 3.0,
                    "department": 2.0,
                    "category": 1.5
                }
            )
        )
    ]
    
    # Create the search index
    index = SearchIndex(
        name=index_name, 
        fields=fields, 
        vector_search=vector_search,
        semantic_search=semantic_search,
        scoring_profiles=scoring_profiles,
        default_scoring_profile="hr-content-scoring"
    )
    
    result = index_client.create_or_update_index(index)
    print(f"Index '{result.name}' created or updated successfully")
    
    return result

if __name__ == "__main__":
    create_hr_documents_index()