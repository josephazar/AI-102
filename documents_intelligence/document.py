"""
ID Document Analysis with Azure Document Intelligence
"""

import os
from .client import get_document_intelligence_client

def analyze_id_document(image_path=None, image_url=None):
    """
    Analyze an identity document (driver's license, passport, etc.) using Document Intelligence
    
    Args:
        image_path (str, optional): Path to a local ID document image
        image_url (str, optional): URL of an ID document image
        
    Returns:
        dict: Structured data extracted from the identity document
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the ID document
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as image:
                poller = client.begin_analyze_document("prebuilt-idDocument", image)
        elif image_url:
            poller = client.begin_analyze_document_from_url(
                "prebuilt-idDocument", 
                image_url
            )
        else:
            return {"error": "No valid image path or URL provided"}
        
        result = poller.result()
        
        if not result.documents or len(result.documents) == 0:
            return {"error": "No ID document found in the image"}
        
        # Extract ID document data
        extracted_data = []
        
        for doc_idx, document in enumerate(result.documents):
            doc_data = {
                "document_index": doc_idx + 1,
                "document_type": document.doc_type if hasattr(document, "doc_type") else "Unknown",
                "confidence": document.confidence,
                "fields": {}
            }
            
            # Map of field names to more readable labels
            field_labels = {
                "FirstName": "first_name",
                "LastName": "last_name",
                "DocumentNumber": "document_number",
                "DateOfBirth": "date_of_birth",
                "DateOfExpiration": "expiration_date",
                "DateOfIssue": "issue_date",
                "DocumentType": "document_type",
                "Sex": "gender",
                "Address": "address",
                "CountryRegion": "country",
                "Region": "state_or_province",
                "MachineReadableZone": "mrz"
            }
            
            # Extract fields from ID document
            for field_name, field in document.fields.items():
                if field_name in field_labels:
                    # Extract field value safely, handling different field types
                    field_value = None
                    
                    # Handle different field content types
                    if hasattr(field, "content"):
                        field_value = field.content
                    elif hasattr(field, "value"):
                        field_value = field.value
                    
                    # Only add if we found a value
                    if field_value is not None:
                        doc_data["fields"][field_labels[field_name]] = {
                            "value": field_value,
                            "confidence": field.confidence if hasattr(field, "confidence") else 0.0
                        }
            
            extracted_data.append(doc_data)
        
        return {"documents": extracted_data}
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Sample ID document URL
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/DriverLicense.png"
    
    # Analyze from URL
    result = analyze_id_document(image_url=sample_url)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for doc in result["documents"]:
            print(f"\n----- ID Document #{doc['document_index']} -----")
            print(f"Document Type: {doc['document_type']}")
            print(f"Confidence: {doc['confidence']:.4f}")
            
            print("\nExtracted Fields:")
            for field_name, field_data in doc["fields"].items():
                print(f"  - {field_name.replace('_', ' ').title()}: {field_data['value']} (Confidence: {field_data['confidence']:.4f})")