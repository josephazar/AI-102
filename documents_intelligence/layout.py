"""
Document Layout Analysis with Azure Document Intelligence
"""

import os
from .client import get_document_intelligence_client

def analyze_document_layout(document_path=None, document_url=None):
    """
    Analyze a document's layout using Document Intelligence
    
    Args:
        document_path (str, optional): Path to a local document
        document_url (str, optional): URL of a document
        
    Returns:
        dict: Structured data about the document layout
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the document
        if document_path and os.path.isfile(document_path):
            with open(document_path, "rb") as f:
                document_content = f.read()
                poller = client.begin_analyze_document(
                    "prebuilt-layout", 
                    document_content
                )
        elif document_url:
            body = {
                "analysisInput": {
                    "source": document_url
                }
            }
            poller = client.begin_analyze_document(
                "prebuilt-layout", 
                body
            )
        else:
            return {"error": "No valid document path or URL provided"}
        
        result = poller.result()
        print(result)
        # Extract overall document information
        document_analysis = {
            "content": result.content,  # Add full document text
            "pages": [],
            "tables": [],
            "paragraphs": [],
            "styles": []
        }
        
        # Extract style information (handwritten vs printed)
        if hasattr(result, "styles") and result.styles is not None:
            for style in result.styles:
                style_data = {
                    "is_handwritten": style.is_handwritten if hasattr(style, "is_handwritten") else False,
                    "confidence": style.confidence if hasattr(style, "confidence") else None
                }
                document_analysis["styles"].append(style_data)
        
        # Extract page information
        for page_idx, page in enumerate(result.pages):
            page_data = {
                "page_number": page_idx + 1,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "text_angle": page.angle,
                "lines": [],
                "words": [],
                "selection_marks": []
            }
            
            # Extract lines
            if hasattr(page, "lines") and page.lines is not None:
                for line_idx, line in enumerate(page.lines):
                    line_data = {
                        "line_number": line_idx + 1,
                        "text": line.content,
                        "bounding_box": line.polygon if hasattr(line, "polygon") else None,
                        "words": []
                    }
                    page_data["lines"].append(line_data)
            
            # Extract words directly from page
            if hasattr(page, "words") and page.words is not None:
                for word_idx, word in enumerate(page.words):
                    word_data = {
                        "word_number": word_idx + 1,
                        "text": word.content,
                        "bounding_box": word.polygon if hasattr(word, "polygon") else None,
                        "confidence": word.confidence if hasattr(word, "confidence") else None
                    }
                    page_data["words"].append(word_data)
            
            # Extract selection marks
            if hasattr(page, "selection_marks") and page.selection_marks is not None:
                for mark_idx, mark in enumerate(page.selection_marks):
                    mark_data = {
                        "mark_number": mark_idx + 1,
                        "state": mark.state if hasattr(mark, "state") else None,
                        "confidence": mark.confidence if hasattr(mark, "confidence") else None,
                        "bounding_box": mark.polygon if hasattr(mark, "polygon") else None
                    }
                    page_data["selection_marks"].append(mark_data)
            
            document_analysis["pages"].append(page_data)
        
        # Extract table information
        if hasattr(result, "tables") and result.tables is not None:
            for table_idx, table in enumerate(result.tables):
                table_data = {
                    "table_number": table_idx + 1,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "bounding_regions": [],
                    "cells": []
                }
                
                # Extract bounding regions
                if hasattr(table, "bounding_regions") and table.bounding_regions is not None:
                    for region in table.bounding_regions:
                        region_data = {
                            "page_number": region.page_number,
                            "polygon": region.polygon
                        }
                        table_data["bounding_regions"].append(region_data)
                
                # Extract cells
                if hasattr(table, "cells") and table.cells is not None:
                    for cell_idx, cell in enumerate(table.cells):
                        cell_data = {
                            "cell_number": cell_idx + 1,
                            "text": cell.content,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "row_span": cell.row_span,
                            "column_span": cell.column_span
                        }
                        table_data["cells"].append(cell_data)
                
                document_analysis["tables"].append(table_data)
        
        # Extract paragraphs if available
        if hasattr(result, "paragraphs") and result.paragraphs is not None:
            for para_idx, paragraph in enumerate(result.paragraphs):
                para_data = {
                    "paragraph_number": para_idx + 1,
                    "text": paragraph.content,
                    "role": paragraph.role if hasattr(paragraph, "role") else None,
                    "bounding_regions": []
                }
                
                if hasattr(paragraph, "bounding_regions") and paragraph.bounding_regions is not None:
                    for region in paragraph.bounding_regions:
                        region_data = {
                            "page_number": region.page_number,
                            "polygon": region.polygon
                        }
                        para_data["bounding_regions"].append(region_data)
                
                document_analysis["paragraphs"].append(para_data)
        
        # Calculate word_count and character_count
        document_analysis["word_count"] = sum(len(page["words"]) for page in document_analysis["pages"])
        document_analysis["character_count"] = len(document_analysis["content"])
        
        return document_analysis
    
    except Exception as e:
        import traceback
        print(f"Error analyzing document layout: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}
    
if __name__ == "__main__":
    # Sample document URL
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    
    # Analyze from URL
    result = analyze_document_layout(document_url=sample_url)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        # Print page information
        print(f"\nDocument has {len(result['pages'])} page(s)")
        print(result['pages'])
        for page in result['pages']:
            print(f"\nPage {page['page_number']} - {page['width']}x{page['height']} {page['unit']}")
            print(f"Text angle: {page['text_angle']} degrees")
            print(f"Lines: {len(page['lines'])}")
            print(f"Words: {len(page['words'])}")
            
            # Print first few lines
            if page['lines']:
                print("\nFirst few lines:")
                for line in page['lines'][:3]:  # Show first 3 lines
                    print(f"  {line['text']}")
                if len(page['lines']) > 3:
                    print("  ...")
        
        # Print table information
        if result['tables']:
            print(f"\nDocument has {len(result['tables'])} table(s)")
            for table in result['tables']:
                print(f"\nTable {table['table_number']} - {table['row_count']}x{table['column_count']} (rows x columns)")
                if table['bounding_regions'] and len(table['bounding_regions']) > 0:
                    print(f"On page {table['bounding_regions'][0]['page_number']}")
                
                # Display a simplified version of the table
                rows = {}
                for cell in table['cells']:
                    row_idx = cell['row_index']
                    col_idx = cell['column_index']
                    if row_idx not in rows:
                        rows[row_idx] = {}
                    rows[row_idx][col_idx] = cell['text']
                
                # Print first few rows
                print("\nTable preview:")
                row_keys = sorted(rows.keys())
                preview_rows = row_keys[:min(3, len(row_keys))]  # Show first 3 rows
                
                for row_idx in preview_rows:
                    row_data = []
                    for col_idx in range(table['column_count']):
                        cell_text = rows[row_idx].get(col_idx, "")
                        if len(cell_text) > 15:
                            cell_text = cell_text[:12] + "..."
                        row_data.append(cell_text)
                    print("  | " + " | ".join(row_data) + " |")
                
                if len(row_keys) > 3:
                    print("  ...")
        
        # Print paragraph information
        if result['paragraphs']:
            print(f"\nDocument has {len(result['paragraphs'])} paragraph(s)")
            preview_paragraphs = result['paragraphs'][:min(3, len(result['paragraphs']))]
            for para in preview_paragraphs:
                para_text = para['text']
                if len(para_text) > 100:
                    para_text = para_text[:97] + "..."
                print(f"\nParagraph {para['paragraph_number']}:")
                print(f"  {para_text}")
                if 'role' in para and para['role']:
                    print(f"  Role: {para['role']}")
                if 'bounding_regions' in para and para['bounding_regions']:
                    print(f"  Page: {para['bounding_regions'][0]['page_number']}")
            if len(result['paragraphs']) > 3:
                print("\n...")