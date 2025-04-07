"""
Utility functions for Azure Document Intelligence demos
"""

import os
import uuid
import requests
from io import BytesIO
import tempfile
from PIL import Image, ImageDraw, ImageFont

def download_sample_files():
    """
    Download sample files for demos if they don't exist
    Returns a dictionary of file paths
    """
    # Create images directory if it doesn't exist
    os.makedirs("documents_intelligence/images", exist_ok=True)
    
    # Sample files URLs and local paths
    samples = {
        "business_card": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/java/ComputerVision/BusinessCard.jpg",
            "path": "documents_intelligence/images/business_card.jpg",
            "type": "Business Card"
        },
        "driver_license": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/DriverLicense.png",
            "path": "documents_intelligence/images/driver_license.png",
            "type": "ID Document"
        },
        "receipt": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/contoso-receipt.png",
            "path": "documents_intelligence/images/receipt.png",
            "type": "Receipt"
        },
        "invoice": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf",
            "path": "documents_intelligence/images/invoice.pdf",
            "type": "Invoice"
        },
        "layout": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf",
            "path": "documents_intelligence/images/layout.pdf",
            "type": "Layout Document"
        },
        "form": {
            "url": "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/master/sdk/formrecognizer/azure-ai-formrecognizer/tests/sample_forms/forms/Form_1.jpg",
            "path": "documents_intelligence/images/form.jpg",
            "type": "Form"
        },
        "income_statement": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-forms/income-statement.png",
            "path": "documents_intelligence/images/income_statement.png",
            "type": "Financial Document"
        },
        "contract": {
            "url": "https://github.com/Azure-Samples/cognitive-services-REST-api-samples/raw/master/curl/form-recognizer/sample-forms/simple-contract.pdf",
            "path": "documents_intelligence/images/contract.pdf",
            "type": "Contract"
        },
        "w2": {
            "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-forms/w2.png",
            "path": "documents_intelligence/images/w2.png",
            "type": "Tax Form"
        }
    }
    
    # Download files that don't exist
    for sample_name, sample_info in samples.items():
        if not os.path.exists(sample_info["path"]):
            try:
                print(f"Downloading {sample_name}...")
                response = requests.get(sample_info["url"])
                response.raise_for_status()
                
                with open(sample_info["path"], "wb") as f:
                    f.write(response.content)
                
                print(f"Downloaded {sample_name} to {sample_info['path']}")
            except Exception as e:
                print(f"Error downloading {sample_name}: {str(e)}")
    
    return samples


def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file and return the file path
    
    Args:
        uploaded_file: Streamlit uploaded file
        
    Returns:
        str: Path to the saved file
    """
    # Create temp directory if it doesn't exist
    os.makedirs("documents_intelligence/temp", exist_ok=True)
    
    # Generate a unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1]
    filename = f"{uuid.uuid4()}{file_extension}"
    filepath = os.path.join("documents_intelligence/temp", filename)
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath


def cleanup_temp_files():
    """
    Remove temporary files older than 1 hour
    """
    import time
    from datetime import datetime, timedelta
    
    temp_dir = "documents_intelligence/temp"
    if not os.path.exists(temp_dir):
        return
    
    # Get current time
    now = time.time()
    one_hour_ago = now - 3600
    
    # Remove files older than 1 hour
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        if os.path.isfile(filepath):
            # Check file creation time
            file_time = os.path.getctime(filepath)
            if file_time < one_hour_ago:
                try:
                    os.remove(filepath)
                    print(f"Removed old temp file: {filepath}")
                except Exception as e:
                    print(f"Error removing temp file {filepath}: {str(e)}")


def visualize_bounding_boxes(image_path, boxes, labels=None, colors=None):
    """
    Create a visualization of an image with bounding boxes
    
    Args:
        image_path (str): Path to the image
        boxes (list): List of bounding boxes, each as [x1, y1, x2, y2] or polygon points
        labels (list, optional): List of labels for each box
        colors (list, optional): List of colors for each box category
        
    Returns:
        BytesIO: Image with bounding boxes as a BytesIO object
    """
    # Open the image
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Default color if none provided
            default_color = (0, 255, 0)
            
            # Try to use a font
            try:
                font = ImageFont.truetype("Arial", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw each bounding box
            for i, box in enumerate(boxes):
                # Get label and color if provided
                label = labels[i] if labels and i < len(labels) else None
                
                if colors and label and label in colors:
                    color = colors[label]
                else:
                    color = default_color
                
                # Draw the box
                if len(box) == 4:  # [x1, y1, x2, y2] format
                    draw.rectangle(box, outline=color, width=2)
                    
                    # Draw label if provided
                    if label:
                        text_position = (box[0], box[1] - 15)
                        # Add a background for the text
                        text_size = draw.textsize(label, font=font) if hasattr(draw, "textsize") else font.getbbox(label)
                        text_width = text_size[2] if isinstance(text_size, tuple) and len(text_size) > 2 else text_size[0]
                        text_height = 15
                        draw.rectangle([text_position[0], text_position[1], text_position[0] + text_width, text_position[1] + text_height], fill=color)
                        draw.text(text_position, label, fill=(255, 255, 255), font=font)
                else:  # Polygon format
                    draw.polygon(box, outline=color, width=2)
                    
                    # Draw label if provided
                    if label:
                        # Position the label at the first point of the polygon
                        text_position = (box[0], box[1] - 15)
                        # Add a background for the text
                        text_size = draw.textsize(label, font=font) if hasattr(draw, "textsize") else font.getbbox(label)
                        text_width = text_size[2] if isinstance(text_size, tuple) and len(text_size) > 2 else text_size[0]
                        text_height = 15
                        draw.rectangle([text_position[0], text_position[1], text_position[0] + text_width, text_position[1] + text_height], fill=color)
                        draw.text(text_position, label, fill=(255, 255, 255), font=font)
            
            # Save the image to a BytesIO object
            output = BytesIO()
            img.save(output, format="PNG")
            output.seek(0)
            
            return output
    except Exception as e:
        print(f"Error visualizing bounding boxes: {str(e)}")
        return None


def convert_pdf_to_image(pdf_path, page_num=0):
    """
    Convert a PDF file to an image
    
    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number to convert (0-based)
        
    Returns:
        str: Path to the image file
    """
    try:
        from pdf2image import convert_from_path
        
        # Create temp directory if it doesn't exist
        os.makedirs("documents_intelligence/temp", exist_ok=True)
        
        # Generate output filename
        output_path = os.path.join("documents_intelligence/temp", f"{uuid.uuid4()}.png")
        
        # Convert PDF page to image
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        
        if images:
            images[0].save(output_path, "PNG")
            return output_path
        else:
            return None
    except Exception as e:
        print(f"Error converting PDF to image: {str(e)}")
        return None


def get_mime_type(file_path):
    """
    Get the MIME type of a file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MIME type of the file
    """
    import mimetypes
    
    # Initialize mimetypes
    mimetypes.init()
    
    # Get the MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return mime_type or "application/octet-stream"