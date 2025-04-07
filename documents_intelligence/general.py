"""
General OCR and Text Extraction with Azure Document Intelligence
"""

import os
from .client import get_document_intelligence_client

def extract_text(document_path=None, document_url=None):
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the document
        if document_path and os.path.isfile(document_path):
            with open(document_path, "rb") as document:
                poller = client.begin_analyze_document("prebuilt-read", document)
        elif document_url:
            poller = client.begin_analyze_document("prebuilt-read", document_url)
        else:
            return {"error": "No valid document path or URL provided"}
        
        result = poller.result()
        
        # Extract overall document information
        text_analysis = {
            "pages": [],
            "content": "",
            "languages": [],  # Changed to "languages" as a list
            "styles": [],
            "word_count": 0,
            "character_count": 0
        }
        
        all_content = []
        word_count = 0
        character_count = 0
        
        # Extract language information
        if hasattr(result, "languages") and result.languages:
            text_analysis["languages"] = [
                {
                    "language_code": lang.language_code,
                    "confidence": lang.confidence
                }
                for lang in result.languages
            ]
        
        # Extract page information (rest of the code remains unchanged)
        for page_idx, page in enumerate(result.pages):
            page_data = {
                "page_number": page_idx + 1,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "angle": page.angle,
                "content": "",
                "lines": [],
                "words": [],
                "selection_marks": []
            }
            
            page_content = []
            
            if page.lines:
                for line_idx, line in enumerate(page.lines):
                    line_data = {
                        "line_number": line_idx + 1,
                        "text": line.content,
                        "bounding_box": line.polygon,
                        "words": []
                    }
                    page_content.append(line.content)
                    page_data["lines"].append(line_data)
            
            if page.words:
                for word_idx, word in enumerate(page.words):
                    word_data = {
                        "word_number": word_idx + 1,
                        "text": word.content,
                        "bounding_box": word.polygon,
                        "confidence": word.confidence
                    }
                    page_data["words"].append(word_data)
                    word_count += 1
                    character_count += len(word.content)
            
            page_data["content"] = "\n".join(page_content)
            text_analysis["pages"].append(page_data)
            all_content.append(page_data["content"])
        
        # Set document content and counts
        text_analysis["content"] = "\n\n".join(all_content)
        text_analysis["word_count"] = word_count or sum(len(page["words"]) for page in text_analysis["pages"])
        text_analysis["character_count"] = character_count or len(text_analysis["content"])
        
        return text_analysis
    
    except Exception as e:
        return {"error": str(e)}
    
def analyze_document(document_path=None, document_url=None, model_id="prebuilt-document"):
    """
    Analyze a document using Document Intelligence's document model
    
    Args:
        document_path (str, optional): Path to a local document
        document_url (str, optional): URL of a document
        model_id (str, optional): The model ID to use (default: prebuilt-document)
        
    Returns:
        dict: Extracted document data including key-value pairs and entities
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the document
        if document_path and os.path.isfile(document_path):
            with open(document_path, "rb") as document:
                poller = client.begin_analyze_document(
                    model_id, 
                    document,
                )
        elif document_url:
            poller = client.begin_analyze_document(
                model_id, 
                document_url
            )
        else:
            return {"error": "No valid document path or URL provided"}
        
        result = poller.result()
        
        # Extract document data
        document_data = {
            "document_type": result.doc_type if hasattr(result, "doc_type") else None,
            "key_value_pairs": [],
            "entities": [],
            "pages": len(result.pages) if hasattr(result, "pages") else 0,
            "tables": []
        }
        
        # Extract key-value pairs
        if hasattr(result, "key_value_pairs") and result.key_value_pairs:
            for kv_idx, kv_pair in enumerate(result.key_value_pairs):
                if kv_pair.key and kv_pair.value:
                    kv_data = {
                        "key": kv_pair.key.content,
                        "value": kv_pair.value.content,
                        "confidence": min(kv_pair.key.confidence, kv_pair.value.confidence)
                    }
                    document_data["key_value_pairs"].append(kv_data)
        
        # Extract document entities
        if hasattr(result, "entities") and result.entities:
            for entity_idx, entity in enumerate(result.entities):
                entity_data = {
                    "category": entity.category,
                    "subcategory": entity.subcategory,
                    "content": entity.content,
                    "confidence": entity.confidence
                }
                document_data["entities"].append(entity_data)
        
        # Extract tables
        if hasattr(result, "tables") and result.tables:
            for table_idx, table in enumerate(result.tables):
                table_data = {
                    "table_number": table_idx + 1,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": []
                }
                
                # Extract cells
                for cell_idx, cell in enumerate(table.cells):
                    cell_data = {
                        "text": cell.content,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "row_span": cell.row_span,
                        "column_span": cell.column_span,
                        "kind": cell.kind if hasattr(cell, "kind") else None,
                        "confidence": cell.confidence
                    }
                    table_data["cells"].append(cell_data)
                
                document_data["tables"].append(table_data)
        
        return document_data
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Sample document URL
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-forms/income-statement.png"
    
    # Extract text from URL
    print("\n--- TEXT EXTRACTION ---")
    text_result = extract_text(document_url=sample_url)
    
    if "error" in text_result:
        print(f"Error: {text_result['error']}")
    else:
        print(f"Language: {text_result['language']}")
        print(f"Pages: {len(text_result['pages'])}")
        print(f"Word count: {text_result['word_count']}")
        print(f"Character count: {text_result['character_count']}")
        
        # Print first page text
        if text_result["pages"]:
            print("\nFirst page content:")
            page_content = text_result["pages"][0]["content"]
            print(page_content[:500] + "..." if len(page_content) > 500 else page_content)
    
    # Analyze document
    print("\n--- DOCUMENT ANALYSIS ---")
    doc_result = analyze_document(document_url=sample_url)
    
    if "error" in doc_result:
        print(f"Error: {doc_result['error']}")
    else:
        print(f"Document type: {doc_result['document_type']}")
        print(f"Pages: {doc_result['pages']}")
        
        # Print key-value pairs
        if doc_result["key_value_pairs"]:
            print("\nKey-Value Pairs:")
            for idx, kv_pair in enumerate(doc_result["key_value_pairs"][:10]):  # Print first 10
                print(f"  {kv_pair['key']}: {kv_pair['value']} (Confidence: {kv_pair['confidence']:.4f})")
            if len(doc_result["key_value_pairs"]) > 10:
                print("  ...")
        
        # Print entities
        if doc_result["entities"]:
            print("\nEntities:")
            for idx, entity in enumerate(doc_result["entities"][:10]):  # Print first 10
                print(f"  {entity['category']}{': ' + entity['subcategory'] if entity['subcategory'] else ''}: {entity['content']} (Confidence: {entity['confidence']:.4f})")
            if len(doc_result["entities"]) > 10:
                print("  ...")
        
        # Print tables
        if doc_result["tables"]:
            print(f"\nTables: {len(doc_result['tables'])}")