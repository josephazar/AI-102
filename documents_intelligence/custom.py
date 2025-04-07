"""
Custom Document Analysis with Azure Document Intelligence
"""

import os
from .client import get_document_intelligence_client

def analyze_custom_document(document_path=None, document_url=None, model_id=None):
    """
    Analyze a document using a custom Document Intelligence model
    
    Args:
        document_path (str, optional): Path to a local document
        document_url (str, optional): URL of a document
        model_id (str): The custom model ID to use
        
    Returns:
        dict: Extracted document data using the custom model
    """
    if not model_id:
        return {"error": "No custom model ID provided"}
    
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the document
        if document_path and os.path.isfile(document_path):
            with open(document_path, "rb") as document:
                poller = client.begin_analyze_document(
                    model_id, 
                    document_stream=document, 
                    content_type="application/octet-stream"
                )
        elif document_url:
            poller = client.begin_analyze_document_from_url(
                model_id, 
                document_url=document_url
            )
        else:
            return {"error": "No valid document path or URL provided"}
        
        result = poller.result()
        
        # Extract document data
        document_data = {
            "model_id": model_id,
            "document_type": result.doc_type if hasattr(result, "doc_type") else None,
            "fields": {},
            "pages": len(result.pages) if hasattr(result, "pages") else 0
        }
        
        # Extract fields
        if hasattr(result, "fields"):
            for field_name, field in result.fields.items():
                if field.value is not None:
                    document_data["fields"][field_name] = {
                        "value": field.value,
                        "confidence": field.confidence
                    }
        
        # Extract document-level confidence if available
        if hasattr(result, "confidence"):
            document_data["confidence"] = result.confidence
        
        return document_data
    
    except Exception as e:
        return {"error": str(e)}


def list_custom_models():
    """
    List all custom models in the Document Intelligence resource
    
    Returns:
        dict: Information about available custom models
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Get all custom models
        models = []
        accounts = client.list_document_models()
        
        for model in accounts:
            model_info = {
                "model_id": model.model_id,
                "description": model.description,
                "created_on": model.created_on,
                "api_version": model.api_version
            }
            models.append(model_info)
        
        return {"models": models}
    
    except Exception as e:
        return {"error": str(e)}


def get_model_details(model_id):
    """
    Get details of a specific Document Intelligence model
    
    Args:
        model_id (str): The model ID to retrieve details for
        
    Returns:
        dict: Detailed information about the model
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Get model details
        model = client.get_document_model(model_id=model_id)
        
        model_details = {
            "model_id": model.model_id,
            "description": model.description,
            "created_on": model.created_on,
            "expires_on": model.expires_on if hasattr(model, "expires_on") else None,
            "api_version": model.api_version,
            "doc_types": {}
        }
        
        # Extract document types
        if hasattr(model, "doc_types"):
            for doc_type, details in model.doc_types.items():
                field_schema = {}
                
                if hasattr(details, "field_schema"):
                    for field_name, field_def in details.field_schema.items():
                        field_schema[field_name] = {
                            "type": field_def.type if hasattr(field_def, "type") else None,
                            "description": field_def.description if hasattr(field_def, "description") else None
                        }
                
                model_details["doc_types"][doc_type] = {
                    "field_schema": field_schema,
                    "field_confidence": details.field_confidence if hasattr(details, "field_confidence") else None
                }
        
        return model_details
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # List all custom models
    print("\n--- CUSTOM MODELS LIST ---")
    models_result = list_custom_models()
    
    if "error" in models_result:
        print(f"Error: {models_result['error']}")
    elif not models_result["models"]:
        print("No custom models found in this resource.")
    else:
        print(f"Found {len(models_result['models'])} custom models:")
        for model in models_result["models"]:
            print(f"  - {model['model_id']} (Created: {model['created_on']})")
            print(f"    Description: {model['description']}")
            print()
        
        # Get details of the first model
        first_model_id = models_result["models"][0]["model_id"]
        print(f"\n--- MODEL DETAILS for {first_model_id} ---")
        details_result = get_model_details(first_model_id)
        
        if "error" in details_result:
            print(f"Error: {details_result['error']}")
        else:
            print(f"Model ID: {details_result['model_id']}")
            print(f"Description: {details_result['description']}")
            print(f"Created: {details_result['created_on']}")
            if details_result["expires_on"]:
                print(f"Expires: {details_result['expires_on']}")
            
            if details_result["doc_types"]:
                print("\nDocument Types:")
                for doc_type, type_details in details_result["doc_types"].items():
                    print(f"  - {doc_type}")
                    
                    if type_details["field_schema"]:
                        print("    Fields:")
                        for field_name, field_info in type_details["field_schema"].items():
                            print(f"      - {field_name} ({field_info['type']})")
                            if field_info["description"]:
                                print(f"        Description: {field_info['description']}")
            
            # Try to analyze a document with this model
            # Add your sample document URL or path here