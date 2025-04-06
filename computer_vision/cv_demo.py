#!/usr/bin/env python
"""
Azure Computer Vision Demo
--------------------------
This script demonstrates how to use Azure Computer Vision services.
It provides a command-line interface for analyzing images using different features.
"""

import os
import time
import json
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import argparse
import sys

# Load environment variables
load_dotenv()

# Get configuration settings
ENDPOINT = os.getenv('COMPUTER_VISION_ENDPOINT')
KEY = os.getenv('COMPUTER_VISION_KEY')
REGION = os.getenv('COMPUTER_VISION_REGION')

def print_colored(text, color="white", bold=False):
    """Print colored text to the console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    style = "\033[1m" if bold else ""
    reset = "\033[0m"
    
    color_code = colors.get(color.lower(), colors["white"])
    print(f"{color_code}{style}{text}{reset}")

def analyze_image(image_path=None, image_url=None, features="caption,read,tags,objects,people"):
    """
    Analyze an image using Azure Computer Vision API.
    
    Args:
        image_path (str, optional): Path to local image file
        image_url (str, optional): URL to remote image
        features (str): Comma-separated list of features to analyze
    
    Returns:
        dict: Analysis results
    """
    if not ENDPOINT or not KEY:
        print_colored("Error: Missing Azure Computer Vision credentials in .env file", "red", True)
        print("Please make sure you have set COMPUTER_VISION_ENDPOINT and COMPUTER_VISION_KEY")
        return None
    
    # API endpoint for image analysis
    analyze_url = f"{ENDPOINT}computervision/imageanalysis:analyze"
    
    # Request parameters
    params = {
        "features": features,
        "model-version": "latest",
        "language": "en",
        "api-version": "2023-10-01"
    }
    
    try:
        if image_path:
            # Read image data for local file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
            # Set headers for binary image data
            headers = {
                "Ocp-Apim-Subscription-Key": KEY,
                "Content-Type": "application/octet-stream"
            }
            
            # Make request with binary image data
            response = requests.post(
                analyze_url,
                headers=headers,
                params=params,
                data=image_data
            )
            
        elif image_url:
            # Set headers for URL-based request
            headers = {
                "Ocp-Apim-Subscription-Key": KEY,
                "Content-Type": "application/json"
            }
            
            # Make request with image URL
            response = requests.post(
                analyze_url,
                headers=headers,
                params=params,
                json={"url": image_url}
            )
        else:
            print_colored("Error: Must provide either image_path or image_url", "red")
            return None
        
        # Check for errors
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print_colored(f"Error making request to Azure Computer Vision API: {str(e)}", "red")
        return None
    except Exception as e:
        print_colored(f"Unexpected error: {str(e)}", "red")
        return None

def save_annotated_image(image_path, result, output_path):
    """
    Create an annotated image with bounding boxes and labels.
    
    Args:
        image_path (str): Path to original image
        result (dict): Analysis results from Computer Vision API
        output_path (str): Path to save the annotated image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the original image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw objects if available
        if 'objectsResult' in result and 'values' in result['objectsResult']:
            for obj in result['objectsResult']['values']:
                # Get bounding box coordinates
                rect = obj.get('boundingBox', {})
                if all(key in rect for key in ['x', 'y', 'w', 'h']):
                    x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
                    
                    # Draw rectangle (orange)
                    draw.rectangle([x, y, x + w, y + h], outline=(255, 165, 0), width=3)
                    
                    # Draw label
                    label = f"{obj.get('name', 'unknown')} ({obj.get('confidence', 0):.2f})"
                    # Draw background for text
                    text_width = max(len(label) * 8, 50)
                    draw.rectangle([x, max(0, y-20), x + text_width, y], fill=(255, 165, 0))
                    draw.text((x + 2, max(0, y - 18)), label, fill=(255, 255, 255), font=font)
        
        # Draw people if available
        if 'peopleResult' in result and 'values' in result['peopleResult']:
            for person in result['peopleResult']['values']:
                # Get bounding box coordinates
                rect = person.get('boundingBox', {})
                if all(key in rect for key in ['x', 'y', 'w', 'h']):
                    x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
                    
                    # Draw rectangle (blue)
                    draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 255), width=3)
                    
                    # Draw label
                    label = f"Person ({person.get('confidence', 0):.2f})"
                    # Draw background for text
                    text_width = max(len(label) * 8, 50)
                    draw.rectangle([x, max(0, y-20), x + text_width, y], fill=(0, 0, 255))
                    draw.text((x + 2, max(0, y - 18)), label, fill=(255, 255, 255), font=font)
        
        # Draw text results if available
        if 'readResult' in result and 'blocks' in result['readResult']:
            for block in result['readResult']['blocks']:
                for line in block.get('lines', []):
                    # Get bounding polygon
                    polygon = line.get('boundingPolygon', [])
                    if len(polygon) >= 4:
                        # Convert to flat list of coordinates
                        points = [(point['x'], point['y']) for point in polygon]
                        
                        # Draw polygon (green)
                        for i in range(len(points)):
                            start = points[i]
                            end = points[(i + 1) % len(points)]
                            draw.line([start, end], fill=(0, 255, 0), width=2)
        
        # Save the annotated image
        image.save(output_path)
        print_colored(f"Annotated image saved to {output_path}", "green")
        return True
        
    except Exception as e:
        print_colored(f"Error creating annotated image: {str(e)}", "red")
        return False

def display_results(result, image_path=None):
    """
    Display the analysis results in a readable format.
    
    Args:
        result (dict): Analysis results from Computer Vision API
        image_path (str, optional): Path to the analyzed image
    """
    if not result:
        print_colored("No results to display", "yellow")
        return
    
    # Print a header
    print("\n" + "="*80)
    print_colored("Azure Computer Vision Analysis Results", "cyan", True)
    print("="*80)
    
    # Display caption if available
    if 'captionResult' in result:
        print_colored("\n📝 Image Caption:", "cyan", True)
        caption = result['captionResult'].get('text', 'No caption available')
        confidence = result['captionResult'].get('confidence', 0)
        print(f"Caption: {caption}")
        print(f"Confidence: {confidence:.2f}")
    
    # Display tags if available
    if 'tagsResult' in result and 'values' in result['tagsResult']:
        print_colored("\n🏷️ Tags:", "cyan", True)
        
        tags = result['tagsResult']['values']
        # Sort tags by confidence
        tags.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Format tags nicely
        for i, tag in enumerate(tags):
            print(f"{i+1:2d}. {tag['name']:15} (Confidence: {tag['confidence']:.2f})")
    
    # Display objects if available
    if 'objectsResult' in result and 'values' in result['objectsResult']:
        print_colored("\n🔍 Objects Detected:", "cyan", True)
        
        objects = result['objectsResult']['values']
        if objects:
            for i, obj in enumerate(objects):
                print(f"{i+1:2d}. {obj['name']:15} (Confidence: {obj['confidence']:.2f})")
                # Print bounding box if available
                if 'boundingBox' in obj:
                    bbox = obj['boundingBox']
                    print(f"    Position: x={bbox.get('x', 0)}, y={bbox.get('y', 0)}, " +
                          f"width={bbox.get('w', 0)}, height={bbox.get('h', 0)}")
        else:
            print("No objects detected")
    
    # Display people if available
    if 'peopleResult' in result and 'values' in result['peopleResult']:
        print_colored("\n👤 People Detected:", "cyan", True)
        
        people = result['peopleResult']['values']
        if people:
            print(f"Found {len(people)} people in the image")
            
            for i, person in enumerate(people):
                print(f"{i+1:2d}. Person (Confidence: {person.get('confidence', 0):.2f})")
                # Print bounding box if available
                if 'boundingBox' in person:
                    bbox = person['boundingBox']
                    print(f"    Position: x={bbox.get('x', 0)}, y={bbox.get('y', 0)}, " +
                          f"width={bbox.get('w', 0)}, height={bbox.get('h', 0)}")
        else:
            print("No people detected")
    
    # Display text (OCR) results if available
    if 'readResult' in result and 'blocks' in result['readResult']:
        print_colored("\n📄 Text Detected (OCR):", "cyan", True)
        
        blocks = result['readResult']['blocks']
        if blocks:
            # Extract all text lines
            all_text = []
            for block in blocks:
                for line in block.get('lines', []):
                    all_text.append(line.get('text', ''))
            
            if all_text:
                for i, line in enumerate(all_text):
                    print(f"{i+1:2d}. {line}")
                
                # Print a summary
                print(f"\nFound {len(blocks)} text blocks with {len(all_text)} lines of text")
            else:
                print("No text content found in detected blocks")
        else:
            print("No text detected")
    
    # If we have an image path, offer to create an annotated version
    if image_path:
        print("\n" + "-"*80)
        create_annotated = input("Would you like to create an annotated image? (y/n): ").lower().strip()
        if create_annotated == 'y':
            # Create output filename
            base_path, ext = os.path.splitext(image_path)
            output_path = f"{base_path}_annotated{ext}"
            
            # Save annotated image
            save_annotated_image(image_path, result, output_path)

def get_sample_images():
    """
    Get a list of sample images from the data folder.
    
    Returns:
        list: List of image file paths
    """
    # Check both possible data folder locations
    data_folders = [
        os.path.join("computer_vision", "data"),  # Module data folder
        "data",  # Root data folder
    ]
    
    for folder in data_folders:
        if os.path.exists(folder):
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_files = [
                os.path.join(folder, f) for f in os.listdir(folder) 
                if os.path.isfile(os.path.join(folder, f)) and 
                any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if image_files:
                return image_files
    
    return []

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Azure Computer Vision Demo")
    
    # Create a mutually exclusive group for image source
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--path", help="Path to local image file")
    group.add_argument("-u", "--url", help="URL to remote image")
    group.add_argument("-s", "--sample", action="store_true", help="Use a sample image")
    
    # Other arguments
    parser.add_argument("-f", "--features", default="caption,read,tags,objects,people",
                      help="Comma-separated list of features to analyze (default: caption,read,tags,objects,people)")
    parser.add_argument("-o", "--output", help="Path to save the JSON results")
    parser.add_argument("-a", "--annotate", action="store_true", help="Create annotated image")
    
    return parser.parse_args()

def main():
    """Main function to run the Azure Computer Vision demo."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print a header
    print("\n" + "="*80)
    print_colored("  Azure Computer Vision Demo", "cyan", True)
    print("="*80)
    
    # Check for required environment variables
    if not ENDPOINT or not KEY:
        print_colored("\nError: Missing Azure Computer Vision credentials in .env file", "red", True)
        print("Please make sure you have set the following environment variables:")
        print("  - COMPUTER_VISION_ENDPOINT")
        print("  - COMPUTER_VISION_KEY")
        return
    
    print(f"\nAzure Computer Vision endpoint: {ENDPOINT}")
    
    # Get image path - from arguments, sample, or user input
    image_path = None
    image_url = None
    
    if args.path:
        image_path = args.path
        if not os.path.exists(image_path):
            print_colored(f"Error: Image file not found: {image_path}", "red")
            return
    elif args.url:
        image_url = args.url
    elif args.sample:
        # Get sample images
        sample_images = get_sample_images()
        
        if not sample_images:
            print_colored("\nError: No sample images found.", "red")
            print("Please run download_images.py first or specify an image file/URL.")
            return
        
        # Let user choose a sample image
        print("\nAvailable sample images:")
        for i, img_path in enumerate(sample_images):
            print(f"{i+1}. {os.path.basename(img_path)}")
        
        # Get user choice
        while True:
            try:
                choice = int(input(f"\nEnter image number (1-{len(sample_images)}): "))
                if 1 <= choice <= len(sample_images):
                    image_path = sample_images[choice-1]
                    break
                else:
                    print_colored(f"Please enter a number between 1 and {len(sample_images)}", "yellow")
            except ValueError:
                print_colored("Please enter a valid number", "yellow")
    else:
        # Interactive mode - ask the user
        image_source = input("\nUse (1) local image file or (2) image URL? Enter 1 or 2: ")
        
        if image_source == "1":
            # Try to get sample images first
            sample_images = get_sample_images()
            
            if sample_images:
                use_sample = input("Would you like to use a sample image? (y/n): ").lower().strip()
                
                if use_sample == 'y':
                    print("\nAvailable sample images:")
                    for i, img_path in enumerate(sample_images):
                        print(f"{i+1}. {os.path.basename(img_path)}")
                    
                    # Get user choice
                    while True:
                        try:
                            choice = int(input(f"\nEnter image number (1-{len(sample_images)}): "))
                            if 1 <= choice <= len(sample_images):
                                image_path = sample_images[choice-1]
                                break
                            else:
                                print_colored(f"Please enter a number between 1 and {len(sample_images)}", "yellow")
                        except ValueError:
                            print_colored("Please enter a valid number", "yellow")
                else:
                    # Ask for file path
                    image_path = input("Enter the path to the image file: ")
                    if not os.path.exists(image_path):
                        print_colored(f"Error: Image file not found: {image_path}", "red")
                        return
            else:
                # No samples available, ask for file path
                image_path = input("Enter the path to the image file: ")
                if not os.path.exists(image_path):
                    print_colored(f"Error: Image file not found: {image_path}", "red")
                    return
        elif image_source == "2":
            image_url = input("Enter the URL of the image: ")
        else:
            print_colored("Invalid option. Please run the script again.", "red")
            return
    
    # Ask for features if not specified
    if not args.features:
        print("\nSelect features to analyze (comma-separated):")
        print("1. Caption (caption)")
        print("2. Text Recognition (read)")
        print("3. Tags & Keywords (tags)")
        print("4. Object Detection (objects)")
        print("5. People Detection (people)")
        
        features_input = input("\nEnter features or press Enter for all: ")
        features = features_input if features_input else "caption,read,tags,objects,people"
    else:
        features = args.features
    
    # Analyze the image
    print("\nAnalyzing image...")
    start_time = time.time()
    
    if image_path:
        print(f"Image: {os.path.basename(image_path)}")
        result = analyze_image(image_path=image_path, features=features)
    else:
        print(f"Image URL: {image_url}")
        result = analyze_image(image_url=image_url, features=features)
    
    elapsed_time = time.time() - start_time
    
    if result:
        print_colored(f"\nAnalysis completed in {elapsed_time:.2f} seconds!", "green")
        
        # Save results to file if requested
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print_colored(f"Results saved to {args.output}", "green")
            except Exception as e:
                print_colored(f"Error saving results: {str(e)}", "red")
        
        # Create annotated image if requested
        if args.annotate and image_path:
            base_path, ext = os.path.splitext(image_path)
            output_path = f"{base_path}_annotated{ext}"
            save_annotated_image(image_path, result, output_path)
        
        # Display the results
        display_results(result, image_path)
    else:
        print_colored("Analysis failed. Please check your inputs and try again.", "red")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nOperation cancelled by user.", "yellow")
        sys.exit(0)