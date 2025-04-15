"""
Azure AI Search - Indexer and Skillset Creation
-----------------------------------------------
This module demonstrates how to create Azure AI Search indexers and skillsets
for processing HR documents from Azure Blob Storage.
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    EntityRecognitionSkill,
    KeyPhraseExtractionSkill,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    SearchIndexerSkillset,
    CognitiveServicesAccountKey,
    SearchIndexer,
    OcrSkill
)

def create_data_source(client, data_source_name="hr-documents-ds"):
    """
    Create a data source connection to Azure Blob Storage.
    
    Args:
        client: The SearchIndexerClient
        data_source_name (str): Name of the data source
        
    Returns:
        The created data source
    """
    connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
    
    container = SearchIndexerDataContainer(name=container_name)
    data_source_connection = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="azureblob",
        connection_string=connection_string,
        container=container
    )
    
    data_source = client.create_or_update_data_source_connection(data_source_connection)
    print(f"Data source '{data_source.name}' created or updated")
    
    return data_source


def create_hr_skillset(client, skillset_name="hr-documents-ss", index_name="hr-documents-index"):
    """
    Create a skillset for processing HR documents.
    
    Args:
        client: The SearchIndexerClient
        skillset_name (str): Name of the skillset
        index_name (str): Name of the target index
        
    Returns:
        The created skillset
    """
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
    cognitive_service_key = os.getenv("AZURE_AI_SERVICE_KEY")
    
    # Define OCR skill for extracting text from images/scanned PDFs
    ocr_skill = OcrSkill(
        description="Extract text from images",
        context="/document/normalized_images/*",
        default_language_code="en",
        should_detect_orientation=True,
        inputs=[
            InputFieldMappingEntry(name="image", source="/document/normalized_images/*")
        ],
        outputs=[
            OutputFieldMappingEntry(name="text", target_name="extracted_text")
        ]
    )
    
    # Define split skill to chunk documents
    split_skill = SplitSkill(
        description="Split skill to chunk documents",
        text_split_mode="pages",
        context="/document",
        maximum_page_length=2000,
        page_overlap_length=500,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/content"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="textItems", target_name="pages")
        ],
    )
    
    # Define embedding skill for vector search
    embedding_skill = AzureOpenAIEmbeddingSkill(
        description="Skill to generate embeddings via Azure OpenAI",
        context="/document/pages/*",
        resource_url=openai_endpoint,
        deployment_name=embedding_model,
        model_name="text-embedding-ada-002",
        dimensions=1536,
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/pages/*"),
        ],
        outputs=[
            OutputFieldMappingEntry(name="embedding", target_name="content_vector")
        ],
    )
    
    # Define entity recognition skill
    entity_skill = EntityRecognitionSkill(
        description="Skill to recognize entities in text",
        context="/document/pages/*",
        categories=["Person", "Organization", "Location", "Email", "URL", "DateTime", "IPAddress"],
        default_language_code="en",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/pages/*")
        ],
        outputs=[
            OutputFieldMappingEntry(name="persons", target_name="persons"),
            OutputFieldMappingEntry(name="organizations", target_name="organizations"),
            OutputFieldMappingEntry(name="locations", target_name="locations"),
            OutputFieldMappingEntry(name="emails", target_name="emails"),
            OutputFieldMappingEntry(name="urls", target_name="urls"),
            OutputFieldMappingEntry(name="dateTimes", target_name="dateTimes")
        ]
    )
    
    # Define key phrase extraction skill
    key_phrase_skill = KeyPhraseExtractionSkill(
        description="Extract key phrases",
        context="/document/pages/*",
        default_language_code="en",
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/pages/*")
        ],
        outputs=[
            OutputFieldMappingEntry(name="keyPhrases", target_name="keyPhrases")
        ]
    )
    
    # Create index projections - fixed by removing parent_id and chunk_id mappings
    index_projections = SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="parent_id",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                    InputFieldMappingEntry(name="content_vector", source="/document/pages/*/content_vector"),
                    InputFieldMappingEntry(name="persons", source="/document/pages/*/persons"),
                    InputFieldMappingEntry(name="organizations", source="/document/pages/*/organizations"),
                    InputFieldMappingEntry(name="locations", source="/document/pages/*/locations"),
                    InputFieldMappingEntry(name="keyPhrases", source="/document/pages/*/keyPhrases"),
                    InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="content", source="/document/content"),
                    InputFieldMappingEntry(name="category", source="/document/metadata_storage_file_extension"),
                    InputFieldMappingEntry(name="last_updated", source="/document/metadata_storage_last_modified"),
                    InputFieldMappingEntry(name="document_type", source="/document/metadata_storage_content_type"),
                    InputFieldMappingEntry(name="department", source="/document/metadata_storage_path"),
                ],
            ),
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
        ),
    )
    
    cognitive_services_account = CognitiveServicesAccountKey(key=cognitive_service_key)
    
    skills = [ocr_skill, split_skill, embedding_skill, entity_skill, key_phrase_skill]
    
    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Skillset to process HR documents",
        skills=skills,
        index_projection=index_projections,
        cognitive_services_account=cognitive_services_account
    )
    
    client.create_or_update_skillset(skillset)
    print(f"Skillset '{skillset_name}' created or updated")
    
    return skillset

def create_indexer(client, indexer_name="hr-documents-idxr", data_source_name="hr-documents-ds", 
                  skillset_name="hr-documents-ss", index_name="hr-documents-index"):
    """
    Create an indexer to process HR documents.
    
    Args:
        client: The SearchIndexerClient
        indexer_name (str): Name of the indexer
        data_source_name (str): Name of the data source
        skillset_name (str): Name of the skillset
        index_name (str): Name of the target index
        
    Returns:
        The created indexer
    """
    # Define indexer parameters for better document processing
    indexer_parameters = {
        "configuration": {
            "dataToExtract": "contentAndMetadata",
            "parsingMode": "default",
            "imageAction": "generateNormalizedImages",
            "allowSkillsetToReadFileData": True,
            "indexStorageMetadataOnlyForOversizedDocuments": True
        },
        "cache": None
    }
    
    # Create the indexer
    indexer = SearchIndexer(
        name=indexer_name,
        description="Indexer for HR documents",
        skillset_name=skillset_name,
        target_index_name=index_name,
        data_source_name=data_source_name,
        parameters=indexer_parameters
    )
    
    result = client.create_or_update_indexer(indexer)
    print(f"Indexer '{result.name}' created or updated")
    
    return result

def setup_hr_document_indexing(data_source_name="hr-documents-ds", skillset_name="hr-documents-ss", 
                              indexer_name="hr-documents-idxr", index_name="hr-documents-index"):
    """
    Set up the full indexing pipeline for HR documents.
    
    Args:
        data_source_name (str): Name of the data source
        skillset_name (str): Name of the skillset
        indexer_name (str): Name of the indexer
        index_name (str): Name of the target index
    """
    # Load environment variables
    load_dotenv()
    
    # Get Azure Search credentials
    search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
    
    # Use API key authentication
    credential = AzureKeyCredential(search_admin_key)
    
    # Create SearchIndexerClient
    indexer_client = SearchIndexerClient(endpoint=search_endpoint, credential=credential)
    
    # Create the pipeline components
    data_source = create_data_source(indexer_client, data_source_name)
    skillset = create_hr_skillset(indexer_client, skillset_name, index_name)
    indexer = create_indexer(indexer_client, indexer_name, data_source_name, skillset_name, index_name)
    
    print(f"HR document indexing pipeline set up successfully!")
    print(f"Indexer '{indexer_name}' is now running. Give it some time to process your documents.")

if __name__ == "__main__":
    setup_hr_document_indexing()