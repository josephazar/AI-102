import streamlit as st
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from io import BytesIO
import time
import base64
import glob
from collections import Counter

# Florence-2 Model Configuration
MODEL_ID = 'microsoft/Florence-2-base'

# Cache for model and processor
@st.cache_resource
def load_florence_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

# Helper Functions
def florence2(model, processor, image, task_prompt, text_input=None):
    """
    Calling the Microsoft Florence2 model
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))

    return parsed_answer

def plot_bbox(image, data):
    """
    Plot BBox for displaying in Streamlit
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1),
                                 x2 - x1,
                                 y2 - y1,
                                 linewidth=2,
                                 edgecolor='lime',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(x1,
                 y1,
                 label,
                 color='black',
                 fontsize=8,
                 bbox=dict(facecolor='lime', alpha=0.8))

    ax.axis('off')
    return fig

def draw_polygons_to_fig(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons and returns figure for Streamlit
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    scale = 1

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = "lime"
        fill_color = "lime" if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_copy)
    ax.axis('off')
    return fig

def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.
    """
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {'bboxes': bboxes, 'labels': labels}
    return od_results

def draw_ocr_to_fig(image, prediction):
    """
    Draw OCR BBox and return figure for Streamlit
    """
    img_copy = image.copy()
    scale = 1
    draw = ImageDraw.Draw(img_copy)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']

    for box, label in zip(bboxes, labels):
        color = 'lime'
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=4, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2),
                  "{}".format(label),
                  align="right",
                  fill=color)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_copy)
    ax.axis('off')
    return fig

def get_image_download_link(img, filename, text):
    """Generate a link to download a PIL image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def show_florence():
    st.title("Florence-2: A Unified Vision Foundation Model")
    
    # Create sidebar for documentation and explanation
    with st.sidebar:
        st.image("florence/assets/florence_logo.png", width=150)
        st.markdown("## About Florence-2")
        st.markdown("""
        Florence-2 is a state-of-the-art vision foundation model that offers a unified approach to various computer vision tasks through simple text prompts.
        
        **Key Features:**
        - Text-prompted image understanding
        - Single model for multiple tasks
        - Zero-shot capabilities
        - Trained on 5.4B visual annotations
        
        [Learn more](https://huggingface.co/papers/2311.06242)
        """)
        
        st.markdown("---")
        st.markdown("### Model Details")
        st.markdown(f"""
        - **Model**: {MODEL_ID}
        - **Provider**: Microsoft
        - **Framework**: PyTorch
        """)
        
        # GitHub link and contribution info
        st.markdown("---")
        st.markdown("### Resources")
        st.markdown("""
        - [Microsoft Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de)
        - [Research Paper](https://huggingface.co/papers/2311.06242)
        """)

    tabs = st.tabs(["Introduction", "Demo", "Use Cases", "Comparison with Azure CV", "Technical Details"])
    
    # Introduction Tab
    with tabs[0]:
        st.header("Understanding Florence-2")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image("florence/florence2_1.jpg", use_container_width=True)
            st.caption("Florence-2 unified architecture for vision tasks")
        
        with col2:
            st.markdown("""
            ### What is Florence-2?
            
            Florence-2 is a powerful vision foundation model that takes a unique approach to visual understanding. It uses a **unified sequence-to-sequence architecture** to handle a variety of vision tasks with simple text prompts.
            
            ### Key Advantages
            
            - **Prompt-based interface**: Simple text instructions to perform complex tasks
            - **Unified representation**: One model for many vision tasks
            - **Comprehensive understanding**: From coarse to fine-grained visual analysis
            - **Pre-trained knowledge**: Built on 5.4 billion visual annotations
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Florence-2 Data Foundation
            
            The model was trained on FLD-5B, a massive dataset comprising:
            
            - 5.4 billion comprehensive visual annotations
            - 126 million images
            - Multiple annotation types per image
            - Iterative strategy of automated image annotation and model refinement
            
            This extensive training enables Florence-2 to develop a deep understanding of visual content and perform a wide range of tasks without task-specific fine-tuning.
            """)
        
        with col2:
            st.image("florence/florence2_2.jpg", use_container_width=True)
            st.caption("Multi-task learning in Florence-2 with varying semantic granularity")
        
        st.markdown("---")
        
        st.subheader("Capabilities")
        
        capability_cols = st.columns(3)
        
        capabilities = [
            {
                "icon": "🖼️",
                "title": "Image Captioning",
                "desc": "Generate detailed descriptions of images with varying levels of detail"
            },
            {
                "icon": "🔍",
                "title": "Object Detection",
                "desc": "Identify and locate objects within images with bounding boxes"
            },
            {
                "icon": "📝",
                "title": "Dense Region Captioning",
                "desc": "Generate descriptions for specific regions in the image"
            },
            {
                "icon": "🔎",
                "title": "Region Proposals",
                "desc": "Suggest regions of interest within the image"
            },
            {
                "icon": "🏷️",
                "title": "Phrase Grounding",
                "desc": "Locate specific phrases or concepts within the image"
            },
            {
                "icon": "🧩",
                "title": "Segmentation",
                "desc": "Create pixel-level masks for objects and regions"
            }
        ]
        
        for i, cap in enumerate(capabilities):
            with capability_cols[i % 3]:
                st.markdown(f"""
                ### {cap['icon']} {cap['title']}
                {cap['desc']}
                """)
    
    # Demo Tab
    with tabs[1]:
        st.header("Florence-2 Interactive Demo")
        
        # Sample image URLs for quick testing
        sample_images = {
            "Car Image": "https://github.com/retkowsky/images/blob/master/car.jpg?raw=true",
            "OCR Image": "https://github.com/retkowsky/images/blob/master/OCR.jpg?raw=true",
            "Car in City Image": "https://github.com/retkowsky/images/blob/master/car%20in%20paris.jpg?raw=true"
        }
        
        # Image upload section
        st.subheader("Step 1: Select an Image")
        
        image_option = st.radio(
            "Choose an image source:",
            ["Upload your own", "Use a sample image", "Provide an image URL"]
        )
        
        image = None
        
        if image_option == "Upload your own":
            uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        elif image_option == "Use a sample image":
            selected_sample = st.selectbox("Choose a sample image:", list(sample_images.keys()))
            if selected_sample:
                img_url = sample_images[selected_sample]
                try:
                    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                    st.image(image, caption=f"Sample Image: {selected_sample}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading sample image: {str(e)}")
        
        elif image_option == "Provide an image URL":
            img_url = st.text_input("Enter the URL of an image:")
            if img_url:
                try:
                    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
                    st.image(image, caption="Image from URL", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image from URL: {str(e)}")
        
        if image:
            # Task Selection
            st.subheader("Step 2: Choose a Vision Task")
            
            task_categories = {
                "Captioning Tasks": [
                    {"name": "Basic Caption", "prompt": "<CAPTION>", "description": "Generate a brief description of the image"},
                    {"name": "Detailed Caption", "prompt": "<DETAILED_CAPTION>", "description": "Generate a more detailed description of the image"},
                    {"name": "Comprehensive Caption", "prompt": "<MORE_DETAILED_CAPTION>", "description": "Generate a comprehensive description with many details"}
                ],
                "Detection Tasks": [
                    {"name": "Object Detection", "prompt": "<OD>", "description": "Detect objects in the image with bounding boxes"},
                    {"name": "Dense Region Caption", "prompt": "<DENSE_REGION_CAPTION>", "description": "Caption specific regions in the image"},
                    {"name": "Region Proposal", "prompt": "<REGION_PROPOSAL>", "description": "Identify regions of interest in the image"}
                ],
                "Interactive Tasks": [
                    {"name": "Phrase Grounding", "prompt": "<CAPTION_TO_PHRASE_GROUNDING>", "description": "Find regions corresponding to specific phrases", "needs_input": True},
                    {"name": "Referring Expression Segmentation", "prompt": "<REFERRING_EXPRESSION_SEGMENTATION>", "description": "Segment regions based on textual descriptions", "needs_input": True},
                    {"name": "Region to Segmentation", "prompt": "<REGION_TO_SEGMENTATION>", "description": "Segment a specified region (format: <loc_x1><loc_y1><loc_x2><loc_y2>)", "needs_input": True},
                    {"name": "Open Vocabulary Detection", "prompt": "<OPEN_VOCABULARY_DETECTION>", "description": "Detect instances of specific concepts", "needs_input": True},
                    {"name": "Region to Category", "prompt": "<REGION_TO_CATEGORY>", "description": "Categorize a specified region (format: <loc_x1><loc_y1><loc_x2><loc_y2>)", "needs_input": True},
                    {"name": "Region to Description", "prompt": "<REGION_TO_DESCRIPTION>", "description": "Describe a specified region (format: <loc_x1><loc_y1><loc_x2><loc_y2>)", "needs_input": True}
                ],
                "OCR Tasks": [
                    {"name": "Text Recognition (OCR)", "prompt": "<OCR>", "description": "Extract text from the image"},
                    {"name": "OCR with Regions", "prompt": "<OCR_WITH_REGION>", "description": "Extract text with location information"}
                ]
            }
            
            task_category = st.selectbox("Select a task category:", list(task_categories.keys()))
            
            if task_category:
                task_options = task_categories[task_category]
                selected_task = st.selectbox(
                    "Select a specific task:",
                    options=task_options,
                    format_func=lambda x: f"{x['name']} - {x['description']}"
                )
                
                # Optional text input for tasks that need it
                text_input = None
                if selected_task.get("needs_input", False):
                    text_input = st.text_input("Enter additional input for this task:", 
                                               help="Some tasks require additional text input. For example, for Phrase Grounding, enter a phrase to locate in the image.")
                
                # Run the analysis
                if st.button("Run Analysis"):
                    with st.spinner("Loading Florence-2 model..."):
                        try:
                            model, processor = load_florence_model()
                            
                            with st.spinner(f"Running {selected_task['name']}..."):
                                # Process the image with Florence-2
                                start_time = time.time()
                                results = florence2(model, processor, image, selected_task["prompt"], text_input)
                                process_time = time.time() - start_time
                                
                                st.success(f"Analysis completed in {process_time:.2f} seconds!")
                                
                                # Display results based on task type
                                st.subheader("Analysis Results")
                                
                                # Special handling for different result types
                                if selected_task["prompt"] in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
                                    # Caption display
                                    st.markdown("### Generated Caption")
                                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; font-size: 18px;">{results}</div>', unsafe_allow_html=True)
                                
                                elif selected_task["prompt"] in ["<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>"]:
                                    # Bounding box visualization
                                    st.markdown(f"### {selected_task['name']} Results")
                                    key = selected_task["prompt"]
                                    if key in results:
                                        fig = plot_bbox(image, results[key])
                                        st.pyplot(fig)
                                        
                                        # Display as table for better visualization
                                        if 'labels' in results[key] and 'bboxes' in results[key]:
                                            detection_data = []
                                            for i, (label, bbox) in enumerate(zip(results[key]['labels'], results[key]['bboxes'])):
                                                detection_data.append({
                                                    "Item": i+1,
                                                    "Label": label,
                                                    "Bounding Box": f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})"
                                                })
                                            
                                            st.dataframe(detection_data)
                                
                                elif selected_task["prompt"] == "<REFERRING_EXPRESSION_SEGMENTATION>":
                                    # Segmentation visualization
                                    st.markdown(f"### Segmentation Results for: '{text_input}'")
                                    key = selected_task["prompt"]
                                    if key in results:
                                        fig = draw_polygons_to_fig(image, results[key], fill_mask=True)
                                        st.pyplot(fig)
                                
                                elif selected_task["prompt"] == "<REGION_TO_SEGMENTATION>":
                                    # Region to segmentation
                                    st.markdown(f"### Segmentation Results for Region: '{text_input}'")
                                    key = selected_task["prompt"]
                                    if key in results:
                                        fig = draw_polygons_to_fig(image, results[key], fill_mask=True)
                                        st.pyplot(fig)
                                
                                elif selected_task["prompt"] == "<OPEN_VOCABULARY_DETECTION>":
                                    # Open vocabulary detection
                                    st.markdown(f"### Detection Results for: '{text_input}'")
                                    key = selected_task["prompt"]
                                    if key in results:
                                        bbox_results = convert_to_od_format(results[key])
                                        fig = plot_bbox(image, bbox_results)
                                        st.pyplot(fig)
                                
                                elif selected_task["prompt"] == "<CAPTION_TO_PHRASE_GROUNDING>":
                                    # Phrase grounding
                                    st.markdown(f"### Location of phrase: '{text_input}'")
                                    key = selected_task["prompt"]
                                    if key in results:
                                        fig = plot_bbox(image, results[key])
                                        st.pyplot(fig)
                                
                                elif selected_task["prompt"] in ["<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"]:
                                    # Region to category/description
                                    st.markdown(f"### Analysis of Region: '{text_input}'")
                                    key = selected_task["prompt"]
                                    if key in results:
                                        st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; font-size: 18px;">{results}</div>', unsafe_allow_html=True)
                                
                                elif selected_task["prompt"] in ["<OCR>", "<OCR_WITH_REGION>"]:
                                    # OCR visualization
                                    st.markdown(f"### Text Recognition Results")
                                    key = selected_task["prompt"]
                                    
                                    if key in results:
                                        if key == "<OCR>":
                                            st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; font-size: 18px;">{results[key]}</div>', unsafe_allow_html=True)
                                        else:
                                            fig = draw_ocr_to_fig(image, results[key])
                                            st.pyplot(fig)
                                            
                                            # Display text results in a clean format
                                            st.markdown("### Extracted Text")
                                            for text in results[key]['labels']:
                                                st.markdown(f"- {text}")
                                
                                # Display raw results in an expander
                                with st.expander("View Raw API Response"):
                                    st.json(results)
                        
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            st.exception(e)
    
    # Use Cases Tab
    with tabs[2]:
        st.header("Florence-2 Use Cases")
        
        use_cases = [
            {
                "title": "E-commerce Product Analysis",
                "icon": "🛍️",
                "description": "Automatically tag and describe products, extract key attributes, identify colors, styles, and materials. Improve product search and discovery, making it easier for customers to find what they're looking for.",
                "tasks": ["Object Detection", "Detailed Captioning", "Product Attribute Extraction"]
            },
            {
                "title": "Content Moderation",
                "icon": "🛡️",
                "description": "Identify potentially sensitive or harmful content in images, flag inappropriate imagery, detect unsafe objects, and ensure platform safety through comprehensive visual understanding.",
                "tasks": ["Object Detection", "Scene Understanding", "Content Classification"]
            },
            {
                "title": "Visual Search",
                "icon": "🔍",
                "description": "Enable users to search for items using images, find similar products based on visual attributes, and bridge the gap between text-based search and visual discovery.",
                "tasks": ["Image Embedding", "Feature Extraction", "Similarity Matching"]
            },
            {
                "title": "Document Processing",
                "icon": "📄",
                "description": "Extract text and structured information from documents, forms, receipts, and business cards, converting unstructured visual data into actionable business intelligence.",
                "tasks": ["OCR", "Layout Analysis", "Form Field Extraction"]
            },
            {
                "title": "Accessibility",
                "icon": "♿",
                "description": "Generate detailed image descriptions for visually impaired users, making digital content more accessible and providing rich context about visual elements.",
                "tasks": ["Detailed Captioning", "Scene Understanding", "Object Relationship Analysis"]
            },
            {
                "title": "Retail Analytics",
                "icon": "📊",
                "description": "Analyze store shelf layouts, track product placement, monitor inventory levels through images, and optimize retail space utilization.",
                "tasks": ["Object Detection", "Product Recognition", "Spatial Analysis"]
            }
        ]
        
        # Display use cases in a grid
        for i in range(0, len(use_cases), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(use_cases):
                    case = use_cases[i + j]
                    with cols[j]:
                        st.markdown(f"""
                        ### {case['icon']} {case['title']}
                        
                        {case['description']}
                        
                        **Key Tasks:**
                        """)
                        for task in case['tasks']:
                            st.markdown(f"- {task}")
                        st.markdown("---")
        
        # Industry Applications
        st.subheader("Industry Applications")
        
        industry_cols = st.columns(3)
        
        industries = [
            {"name": "Healthcare", "icon": "🏥", "example": "Assist in medical image analysis and patient record digitization"},
            {"name": "Manufacturing", "icon": "🏭", "example": "Visual inspection and quality control of products"},
            {"name": "Real Estate", "icon": "🏠", "example": "Automatic property image categorization and feature extraction"},
            {"name": "Agriculture", "icon": "🌾", "example": "Crop monitoring and disease detection from aerial imagery"},
            {"name": "Automotive", "icon": "🚗", "example": "Vehicle damage assessment and part identification"},
            {"name": "Media & Entertainment", "icon": "🎬", "example": "Content tagging, scene analysis, and media organization"}
        ]
        
        for i, industry in enumerate(industries):
            with industry_cols[i % 3]:
                st.markdown(f"""
                ### {industry['icon']} {industry['name']}
                {industry['example']}
                """)
    
    # Comparison with Azure CV
    with tabs[3]:
        st.header("Florence-2 vs. Azure Computer Vision")
        
        st.markdown("""
        While both Florence-2 and Azure Computer Vision provide powerful image analysis capabilities, 
        they have different approaches and strengths. Understanding these differences can help you 
        choose the right solution for your needs.
        """)
        
        comparison_data = {
            "Architecture": [
                "Unified sequence-to-sequence model for all vision tasks",
                "Suite of specialized models for different vision tasks"
            ],
            "Interface": [
                "Text prompt-based approach with natural language instructions",
                "Structured API calls with predefined parameters"
            ],
            "Flexibility": [
                "High flexibility with open-ended prompting",
                "Predefined capabilities with consistent, reliable outputs"
            ],
            "Deployment": [
                "Can be accessed via Hugging Face and research frameworks",
                "Fully integrated into Azure cloud ecosystem with enterprise support"
            ],
            "Use Cases": [
                "Research, exploration, and custom vision applications",
                "Production enterprise applications with SLAs and compliance"
            ],
            "Training Data": [
                "FLD-5B dataset with 5.4B annotations on 126M images",
                "Microsoft's proprietary datasets, continuously updated"
            ]
        }
        
        # Create the comparison table
        for feature, values in comparison_data.items():
            cols = st.columns([1, 2, 2])
            with cols[0]:
                st.markdown(f"### {feature}")
            with cols[1]:
                st.markdown(f"""
                #### Florence-2
                {values[0]}
                """)
            with cols[2]:
                st.markdown(f"""
                #### Azure Computer Vision
                {values[1]}
                """)
            st.markdown("---")
        
        # When to use which
        st.subheader("When to Use Each Service")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### When to use Florence-2
            
            - Research and experimentation with cutting-edge vision AI
            - Applications needing flexible, open-ended visual understanding
            - Projects requiring fine-grained control over prompts and outputs
            - Use cases benefiting from unified multi-task learning
            - Development environments where cloud API integration isn't required
            """)
        
        with col2:
            st.markdown("""
            ### When to use Azure Computer Vision
            
            - Enterprise production applications requiring SLAs
            - Projects needing seamless integration with Azure ecosystem
            - Applications requiring compliance with Microsoft's enterprise standards
            - Use cases needing consistent, well-documented API responses
            - Solutions requiring specialized vision features with enterprise support
            """)
    
    # Technical Details Tab
    with tabs[4]:
        st.header("Technical Deep Dive")
        
        st.markdown("""
        ### Florence-2 Architecture
        
        Florence-2 employs a sequence-to-sequence (Seq2Seq) architecture, which unifies visual understanding 
        across different tasks. This differs from traditional computer vision approaches that use separate 
        models for different tasks.
        
        The model consists of:
        
        1. **Vision Encoder**: Transforms the image into a sequence of visual tokens
        2. **Text Encoder**: Encodes the text prompt into a sequence of text tokens
        3. **Cross-Modal Fusion**: Merges visual and textual information
        4. **Decoder**: Generates the output based on the fused representation
        """)
        
        st.markdown("""
        ### Prompt Engineering for Florence-2
        
        Florence-2 uses a special prompt format with task-specific tokens enclosed in angle brackets. 
        Here are some examples:
        
        - `<CAPTION>` - Generate a basic caption for an image
        - `<OD>` - Perform object detection
        - `<REFERRING_EXPRESSION_SEGMENTATION>` + text - Segment an object described by the text
        
        These special tokens help the model understand what type of vision task is being requested.
        """)
        
        st.markdown("""
        ### Model Variants
        
        The Florence-2 family includes multiple model sizes:
        
        | Model | Parameters | Description |
        | --- | --- | --- |
        | Florence-2-small | ~2B | Smaller, faster variant with good performance |
        | Florence-2-base | ~5B | Balanced performance and efficiency |
        | Florence-2-large | ~13B | Highest performance for complex tasks |
        
        This demo uses the **base** variant, which offers a good balance of performance and efficiency.
        """)
        
        st.markdown("""
        ### Implementation Details
        
        To use Florence-2 in your own applications:
        
        ```python
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Load model and processor
        model_id = 'microsoft/Florence-2-base'
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Process an image with a prompt
        prompt = "<CAPTION>"  # Or any other task prompt
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Generate output
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        ```
        
        ### Runtime Requirements
        
        - Python 3.7+
        - PyTorch 1.10+
        - Transformers 4.26.0+
        - 16GB+ RAM recommended (varies by model size)
        - GPU acceleration strongly recommended
        - For local deployment, install the required dependencies using:
        
        ```bash
        pip install -r requirements.txt
        ```
        """)
if __name__ == "__main__":
    # Run the Streamlit app
    show_florence()