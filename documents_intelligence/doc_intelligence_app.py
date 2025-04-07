"""
Document Intelligence Streamlit Application
"""

import os
import streamlit as st
import pandas as pd
import json
import time
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import Document Intelligence modules
from documents_intelligence.client import get_document_intelligence_client
from documents_intelligence.business_card import analyze_business_card
from documents_intelligence.document import analyze_id_document
from documents_intelligence.receipt import analyze_receipt
from documents_intelligence.invoice import analyze_invoice
from documents_intelligence.layout import analyze_document_layout
from documents_intelligence.general import extract_text, analyze_document
from documents_intelligence.utils import (
    download_sample_files, save_uploaded_file, cleanup_temp_files, 
    visualize_bounding_boxes, convert_pdf_to_image, get_mime_type
)

# Global variables
SAMPLE_FILES = None


def show_document_intelligence():
    """
    Main function to show the Document Intelligence Streamlit application
    """
    global SAMPLE_FILES
    
    # Set up the page
    st.title("Azure Document Intelligence Explorer")
    st.markdown("""
    <style>
    .document-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .document-field {
        padding: 10px;
        background-color: white;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #0078d4;
    }
    .highlight-box {
        border: 2px dashed #0078d4;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: rgba(0, 120, 212, 0.1);
    }
    .success-box {
        border-left: 4px solid #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .info-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create sidebar for navigation
    st.sidebar.title("Document Intelligence")
    
    # Download sample files if not already downloaded
    if SAMPLE_FILES is None:
        with st.spinner("Loading sample files..."):
            SAMPLE_FILES = download_sample_files()
    
    # Navigation options
    nav_options = [
        "Overview",
        "ID Document Analysis",
        "Receipt Analysis",
        "Invoice Processing",
        "Document Layout Analysis",
        "General Document Analysis",
    ]
    
    # User authentication status
    client_initialized = get_document_intelligence_client() is not None
    if not client_initialized:
        st.sidebar.error("Document Intelligence client not initialized. Please check your credentials in .env file.")
        nav_options = ["Overview"]
    
    # Navigation
    selected_option = st.sidebar.radio("Select Feature", nav_options)
    
    # Show appropriate page based on selection
    if selected_option == "Overview":
        show_overview_page()
    elif selected_option == "ID Document Analysis":
        show_id_document_page()
    elif selected_option == "Receipt Analysis":
        show_receipt_page()
    elif selected_option == "Invoice Processing":
        show_invoice_page()
    elif selected_option == "Document Layout Analysis":
        show_layout_page()
    elif selected_option == "General Document Analysis":
        show_general_document_page()

    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This demo application showcases Azure Document Intelligence capabilities. "
        "For more information, visit the [Microsoft Learn documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)."
    )


def show_overview_page():
    """
    Display the overview page with feature descriptions
    """
    st.header("Azure Document Intelligence")
    
    st.markdown("""
    <div class="info-card">
    <h3>Transform Documents into Actionable Data</h3>
    <p>Azure Document Intelligence (formerly Form Recognizer) applies advanced machine learning to extract text, key-value pairs, tables, and structure from documents. Turn your documents into usable data in seconds instead of hours.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    ## Key Benefits
    
    - **Accuracy**: Industry-leading OCR and document understanding
    - **Customization**: Adapt to your specific documents with minimal training
    - **Integration**: Process documents in your apps, workflows, and business processes
    - **Security**: Azure enterprise-grade security and compliance
    
    ## Core Capabilities
    """)
    
    # Feature cards in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="document-card">
        <h3>📇 Business Card Analysis</h3>
        <p>Extract contact information, job titles, companies, phone numbers, emails, and more from business cards automatically.</p>
        </div>
        
        <div class="document-card">
        <h3>🧾 Receipt Processing</h3>
        <p>Analyze receipts to extract merchant details, transaction information, line items, and totals in structured data format.</p>
        </div>
        
        <div class="document-card">
        <h3>📄 Document Layout</h3>
        <p>Understand document structure by extracting text, tables, selection marks, and organizing content with spatial information.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="document-card">
        <h3>🪪 ID Document Analysis</h3>
        <p>Process driver's licenses, passports, and other ID documents to extract personal information in a secure manner.</p>
        </div>
        
        <div class="document-card">
        <h3>📋 Invoice Processing</h3>
        <p>Turn invoices into structured data by extracting vendor information, customer details, line items, and payment information.</p>
        </div>
        
        <div class="document-card">
        <h3>💼 Custom Document Processing</h3>
        <p>Train on your specific document types with as few as 5 samples to create custom document processing models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Industry solutions
    st.markdown("""
    ## Industry Solutions
    
    Document Intelligence can transform business processes across multiple industries:
    """)
    
    industries = [
        {
            "name": "Financial Services",
            "description": "Automate loan processing, check deposit verification, insurance claims, and financial statement analysis",
            "icon": "💰"
        },
        {
            "name": "Healthcare",
            "description": "Extract data from patient forms, medical records, insurance cards, and prescription documents",
            "icon": "🏥"
        },
        {
            "name": "Legal",
            "description": "Analyze contracts, legal briefs, case files, and supporting documentation",
            "icon": "⚖️"
        },
        {
            "name": "Retail & E-commerce",
            "description": "Process purchase orders, shipping documents, and inventory records",
            "icon": "🛍️"
        },
        {
            "name": "Manufacturing",
            "description": "Extract information from bills of materials, service manuals, and compliance documents",
            "icon": "🏭"
        },
        {
            "name": "Government",
            "description": "Process tax forms, benefit applications, permits, and regulatory documents",
            "icon": "🏛️"
        }
    ]
    
    # Industry columns
    cols = st.columns(3)
    for i, industry in enumerate(industries):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h3>{industry['icon']} {industry['name']}</h3>
            <p>{industry['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ROI and Value Proposition
    st.markdown("""
    ## The Business Value
    
    Implementing Azure Document Intelligence can deliver significant return on investment:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Time Savings", "70-80%", "↓ in document processing time")
    
    with col2:
        st.metric("Cost Reduction", "35-45%", "↓ in operational costs")
    
    with col3:
        st.metric("Accuracy Improvement", "90-99%", "↑ in data extraction accuracy")
    
    # Call-to-action
    st.markdown("""
    <div class="highlight-box">
    <h3>Ready to See It in Action?</h3>
    <p>Select a feature from the sidebar to explore Azure Document Intelligence capabilities with your own documents or our sample files.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add integration information
    st.markdown("""
    ## Integration Options
    
    Azure Document Intelligence can be integrated with your systems in multiple ways:
    
    - **REST API**: Direct HTTP calls with JSON responses
    - **Client Libraries**: SDKs for Python, .NET, Java, JavaScript, and Go
    - **Power Automate**: Pre-built connectors for low-code workflows
    - **Logic Apps**: Serverless workflow integration
    - **Azure Form Recognizer Studio**: Web-based UI for training and testing
    
    """)



def show_id_document_page():
    """
    Display the ID document analysis page
    """
    st.header("ID Document Analysis")
    
    st.markdown("""
    Extract information from identity documents like driver's licenses and ID cards. 
    Upload an ID document image or use our sample.
    
    ### Key Capabilities
    
    - Extract personal information (name, DOB, etc.)
    - Identify document numbers and expiration dates
    - Detect addresses and other details
    - Works with driver's licenses, ID cards, and passports
    """)
    
    # Let user upload a file or use sample
    option = st.radio(
        "Choose an option",
        ["Upload my own ID document", "Use a sample ID document"]
    )
    
    image_path = None
    image_url = None
    
    if option == "Upload my own ID document":
        uploaded_file = st.file_uploader("Upload an ID document image", type=["jpg", "jpeg", "png", "gif", "bmp"])
        if uploaded_file:
            image_path = save_uploaded_file(uploaded_file)
            st.image(image_path, caption="Uploaded ID Document", use_container_width=True)
    else:
        sample_path = SAMPLE_FILES["driver_license"]["path"]
        st.image(sample_path, caption="Sample ID Document", use_container_width=True)
        image_path = sample_path
    
    # Process the ID document
    if image_path and st.button("Extract ID Information"):
        with st.spinner("Analyzing ID document..."):
            start_time = time.time()
            result = analyze_id_document(image_path=image_path)
            processing_time = time.time() - start_time
            
            if "error" in result:
                st.error(f"Error analyzing ID document: {result['error']}")
            else:
                st.success(f"Analysis completed in {processing_time:.2f} seconds!")
                
                # Display the extracted data
                for i, doc in enumerate(result["documents"]):
                    st.markdown(f"""
                    <div class="document-card">
                    <h3>ID Document #{i+1}</h3>
                    <p>Document Type: {doc['document_type']}</p>
                    <p>Confidence: {doc['confidence']:.2%}</p>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different sections
                    tabs = st.tabs(["Personal Information", "Document Details", "Visualization", "Raw Data"])
                    
                    with tabs[0]:
                        st.markdown("<h4>Personal Information</h4>", unsafe_allow_html=True)
                        
                        # Get personal info fields
                        personal_fields = {k: v for k, v in doc["fields"].items() 
                                         if k in ["first_name", "last_name", "date_of_birth", "gender", "address"]}
                        
                        if personal_fields:
                            for field_name, field_data in personal_fields.items():
                                st.markdown(f"""
                                <div class="document-field">
                                <p><strong>{field_name.replace('_', ' ').title()}:</strong> {field_data['value']}</p>
                                <p style="font-size: 0.8em; color: #666;">Confidence: {field_data['confidence']:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No personal information found.")
                    
                    with tabs[1]:
                        st.markdown("<h4>Document Details</h4>", unsafe_allow_html=True)
                        
                        # Get document info fields
                        doc_fields = {k: v for k, v in doc["fields"].items() 
                                    if k in ["document_number", "expiration_date", "issue_date", 
                                           "document_type", "country", "state_or_province"]}
                        
                        if doc_fields:
                            for field_name, field_data in doc_fields.items():
                                st.markdown(f"""
                                <div class="document-field">
                                <p><strong>{field_name.replace('_', ' ').title()}:</strong> {field_data['value']}</p>
                                <p style="font-size: 0.8em; color: #666;">Confidence: {field_data['confidence']:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No document details found.")
                    
                    with tabs[2]:
                        st.subheader("Data Extraction Visualization")
                        
                        # Create a confidence chart for all fields
                        fields = []
                        confidences = []
                        
                        for field_name, field_data in doc["fields"].items():
                            fields.append(field_name.replace('_', ' ').title())
                            confidences.append(field_data['confidence'])
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            "Field": fields,
                            "Confidence": confidences
                        })
                        
                        # Sort by confidence
                        df = df.sort_values("Confidence", ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            df,
                            x="Confidence",
                            y="Field",
                            title="Field Extraction Confidence",
                            orientation='h',
                            color="Confidence",
                            color_continuous_scale="blues",
                            range_color=[0, 1]
                        )
                        
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tabs[3]:
                        st.subheader("Raw Extracted Data")
                        st.json(doc)
                    
                    st.markdown("</div>", unsafe_allow_html=True)


def show_receipt_page():
    """
    Display the receipt analysis page
    """
    st.header("Receipt Analysis")
    
    st.markdown("""
    Extract detailed information from receipts. Upload a receipt image or use our sample.
    
    ### Key Capabilities
    
    - Identify merchant information
    - Extract transaction details (date, time, total)
    - Itemize purchases with quantities and prices
    - Detect tax, subtotal, tip amounts
    """)
    
    # Let user upload a file or use sample
    option = st.radio(
        "Choose an option",
        ["Upload my own receipt", "Use a sample receipt"]
    )
    
    image_path = None
    
    if option == "Upload my own receipt":
        uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png", "gif", "bmp"])
        if uploaded_file:
            image_path = save_uploaded_file(uploaded_file)
            st.image(image_path, caption="Uploaded Receipt", use_container_width=True)
    else:
        sample_path = SAMPLE_FILES["receipt"]["path"]
        st.image(sample_path, caption="Sample Receipt", use_container_width=True)
        image_path = sample_path
    
    # Process the receipt
    if image_path and st.button("Extract Receipt Information"):
        with st.spinner("Analyzing receipt..."):
            start_time = time.time()
            result = analyze_receipt(image_path=image_path)
            processing_time = time.time() - start_time
            
            if "error" in result:
                st.error(f"Error analyzing receipt: {result['error']}")
            else:
                st.success(f"Analysis completed in {processing_time:.2f} seconds!")
                
                # Display the extracted data
                for i, receipt in enumerate(result["receipts"]):
                    st.markdown(f"""
                    <div class="document-card">
                    <h3>Receipt #{i+1}</h3>
                    <p>Confidence: {receipt['confidence']:.2%}</p>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different sections
                    tabs = st.tabs(["Transaction Details", "Items", "Payment Info", "Visualization", "Raw Data"])
                    
                    with tabs[0]:
                        st.markdown("<h4>Transaction Details</h4>", unsafe_allow_html=True)
                        
                        # Display merchant info
                        if receipt["merchant"]:
                            st.markdown(f"""
                            <div class="document-field">
                            <h4>🏬 {receipt["merchant"]}</h4>
                            """, unsafe_allow_html=True)
                            
                            # Contact info
                            contact_sections = []
                            
                            if receipt["contact_info"]["phone"]:
                                contact_sections.append(f"<strong>Phone:</strong> {receipt['contact_info']['phone']}")
                            
                            if receipt["contact_info"]["address"]:
                                contact_sections.append(f"<strong>Address:</strong> {receipt['contact_info']['address']}")
                            
                            if receipt["contact_info"]["merchant_url"]:
                                contact_sections.append(f"<strong>Website:</strong> {receipt['contact_info']['merchant_url']}")
                            
                            if contact_sections:
                                st.markdown("<p>" + "<br>".join(contact_sections) + "</p>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Transaction details
                        st.markdown(f"""
                        <div class="document-field">
                        <h4>🧾 Transaction</h4>
                        """, unsafe_allow_html=True)
                        
                        transaction_sections = []
                        
                        if receipt["transaction"]["date"]:
                            transaction_sections.append(f"<strong>Date:</strong> {receipt['transaction']['date']}")
                        
                        if receipt["transaction"]["time"]:
                            transaction_sections.append(f"<strong>Time:</strong> {receipt['transaction']['time']}")
                        
                        if receipt["transaction"]["total"]:
                            transaction_sections.append(f"<strong>Total:</strong> {receipt['transaction']['total']}")
                        
                        if receipt["transaction"]["subtotal"]:
                            transaction_sections.append(f"<strong>Subtotal:</strong> {receipt['transaction']['subtotal']}")
                        
                        if receipt["transaction"]["tax"]:
                            transaction_sections.append(f"<strong>Tax:</strong> {receipt['transaction']['tax']}")
                        
                        if receipt["transaction"]["tip"]:
                            transaction_sections.append(f"<strong>Tip:</strong> {receipt['transaction']['tip']}")
                        
                        if transaction_sections:
                            st.markdown("<p>" + "<br>".join(transaction_sections) + "</p>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p>No transaction details found.</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with tabs[1]:
                        st.markdown("<h4>Items</h4>", unsafe_allow_html=True)
                        
                        if receipt["items"]:
                            # Create a table of items
                            items_data = []
                            
                            for item in receipt["items"]:
                                item_data = {
                                    "Name": item.get("name", "Unnamed Item"),
                                    "Quantity": item.get("quantity", ""),
                                    "Price": item.get("price", ""),
                                    "Total": item.get("total_price", "")
                                }
                                items_data.append(item_data)
                            
                            # Display as a DataFrame
                            df = pd.DataFrame(items_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Calculate summary
                            item_count = len(receipt["items"])
                            # Replace the problematic line in show_receipt_page() with:
                            total_quantity = sum(float(item.get("quantity", 0)) for item in receipt["items"])
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Number of Items", item_count)
                            col2.metric("Total Quantity", f"{total_quantity:.2f}" if total_quantity > 0 else "N/A")
                        else:
                            st.info("No items found on the receipt.")
                    
                    with tabs[2]:
                        st.markdown("<h4>Payment Information</h4>", unsafe_allow_html=True)
                        
                        # Payment info
                        payment_info = receipt["payment_info"]
                        
                        if any(payment_info.values()):
                            st.markdown(f"""
                            <div class="document-field">
                            <h4>💳 Payment Details</h4>
                            """, unsafe_allow_html=True)
                            
                            payment_sections = []
                            
                            if payment_info["card_type"]:
                                payment_sections.append(f"<strong>Card Type:</strong> {payment_info['card_type']}")
                            
                            if payment_info["card_number"]:
                                payment_sections.append(f"<strong>Card Number:</strong> {payment_info['card_number']}")
                            
                            if payment_sections:
                                st.markdown("<p>" + "<br>".join(payment_sections) + "</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p>No payment details found.</p>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("No payment information found.")
                    
                    with tabs[3]:
                        st.subheader("Receipt Visualization")
                        
                        # Pie chart of costs
                        cost_data = {
                            "Subtotal": receipt["transaction"]["subtotal"] if receipt["transaction"]["subtotal"] else 0,
                            "Tax": receipt["transaction"]["tax"] if receipt["transaction"]["tax"] else 0,
                            "Tip": receipt["transaction"]["tip"] if receipt["transaction"]["tip"] else 0
                        }
                        
                        # Check if we have actual values and convert to numeric
                        valid_costs = {}
                        for category, value in cost_data.items():
                            if value:
                                try:
                                    if isinstance(value, str):
                                        # Remove currency symbols and commas
                                        cleaned_value = value.replace('$', '').replace(',', '')
                                        valid_costs[category] = float(cleaned_value)
                                    else:
                                        valid_costs[category] = float(value)
                                except (ValueError, TypeError):
                                    pass
                        
                        if valid_costs:
                            # Create a DataFrame
                            df = pd.DataFrame({
                                "Category": list(valid_costs.keys()),
                                "Amount": list(valid_costs.values())
                            })
                            
                            # Create a pie chart
                            fig = px.pie(
                                df,
                                values="Amount",
                                names="Category",
                                title="Receipt Breakdown",
                                color_discrete_sequence=px.colors.sequential.Blues_r
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Items bar chart
                        if receipt["items"]:
                            items_with_price = []
                            
                            for item in receipt["items"]:
                                if "name" in item and "total_price" in item:
                                    try:
                                        if isinstance(item["total_price"], str):
                                            # Remove currency symbols and commas
                                            cleaned_price = item["total_price"].replace('$', '').replace(',', '')
                                            price = float(cleaned_price)
                                        else:
                                            price = float(item["total_price"])
                                        
                                        items_with_price.append({
                                            "name": item["name"] if len(item["name"]) < 20 else item["name"][:17] + "...",
                                            "price": price
                                        })
                                    except (ValueError, TypeError):
                                        pass
                            
                            if items_with_price:
                                # Create a DataFrame
                                df = pd.DataFrame(items_with_price)
                                
                                # Sort by price
                                df = df.sort_values("price", ascending=False)
                                
                                # Limit to top 10 items
                                if len(df) > 10:
                                    df = df.head(10)
                                
                                # Create a bar chart
                                fig = px.bar(
                                    df,
                                    x="name",
                                    y="price",
                                    title="Item Prices",
                                    color="price",
                                    color_continuous_scale="blues"
                                )
                                
                                fig.update_layout(xaxis_title="Item", yaxis_title="Price")
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with tabs[4]:
                        st.subheader("Raw Extracted Data")
                        st.json(receipt)
                    
                    st.markdown("</div>", unsafe_allow_html=True)


def show_invoice_page():
    """
    Display the invoice processing page
    """
    st.header("Invoice Processing")
    
    st.markdown("""
    Extract detailed information from invoices. Upload an invoice document or use our sample.
    
    ### Key Capabilities
    
    - Extract vendor and customer information
    - Identify invoice numbers, dates, and payment terms
    - Itemize purchases with detailed line items
    - Calculate totals, tax, and payment information
    """)
    
    # Let user upload a file or use sample
    option = st.radio(
        "Choose an option",
        ["Upload my own invoice", "Use a sample invoice"]
    )
    
    document_path = None
    
    if option == "Upload my own invoice":
        uploaded_file = st.file_uploader("Upload an invoice document", type=["pdf", "jpg", "jpeg", "png", "tiff"])
        if uploaded_file:
            document_path = save_uploaded_file(uploaded_file)
            try:
                # Try to display the document
                if document_path.lower().endswith('.pdf'):
                    # For PDFs, convert the first page to an image for display
                    image_path = convert_pdf_to_image(document_path)
                    if image_path:
                        st.image(image_path, caption="Uploaded Invoice (First Page)", use_container_width=True)
                    else:
                        st.warning("Unable to display PDF preview. Analysis will still work.")
                else:
                    st.image(document_path, caption="Uploaded Invoice", use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying document: {str(e)}")
    else:
        sample_path = SAMPLE_FILES["invoice"]["path"]
        # For PDFs, convert the first page to an image for display
        if sample_path.lower().endswith('.pdf'):
            image_path = convert_pdf_to_image(sample_path)
            if image_path:
                st.image(image_path, caption="Sample Invoice (First Page)", use_container_width=True)
        else:
            st.image(sample_path, caption="Sample Invoice", use_container_width=True)
        document_path = sample_path
    
    # Process the invoice
    if document_path and st.button("Extract Invoice Information"):
        with st.spinner("Analyzing invoice..."):
            start_time = time.time()
            result = analyze_invoice(document_path=document_path)
            processing_time = time.time() - start_time
            
            if "error" in result:
                st.error(f"Error analyzing invoice: {result['error']}")
            else:
                st.success(f"Analysis completed in {processing_time:.2f} seconds!")
                
                # Display the extracted data
                for i, invoice in enumerate(result["invoices"]):
                    st.markdown(f"""
                    <div class="document-card">
                    <h3>Invoice #{i+1}</h3>
                    <p>Confidence: {invoice['confidence']:.2%}</p>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different sections
                    tabs = st.tabs(["Invoice Details", "Vendor & Customer", "Line Items", "Visualization", "Raw Data"])
                    
                    with tabs[0]:
                        st.markdown("<h4>Invoice Details</h4>", unsafe_allow_html=True)
                        
                        # Invoice details
                        invoice_details = invoice["invoice_details"]
                        
                        if any(invoice_details.values()):
                            st.markdown(f"""
                            <div class="document-field">
                            <h4>📄 Invoice Information</h4>
                            """, unsafe_allow_html=True)
                            
                            details_sections = []
                            
                            if invoice_details["id"]:
                                details_sections.append(f"<strong>Invoice #:</strong> {invoice_details['id']}")
                            
                            if invoice_details["date"]:
                                details_sections.append(f"<strong>Invoice Date:</strong> {invoice_details['date']}")
                            
                            if invoice_details["due_date"]:
                                details_sections.append(f"<strong>Due Date:</strong> {invoice_details['due_date']}")
                            
                            if invoice_details["purchase_order"]:
                                details_sections.append(f"<strong>PO Number:</strong> {invoice_details['purchase_order']}")
                            
                            if invoice_details["service_start_date"]:
                                details_sections.append(f"<strong>Service Start Date:</strong> {invoice_details['service_start_date']}")
                            
                            if invoice_details["service_end_date"]:
                                details_sections.append(f"<strong>Service End Date:</strong> {invoice_details['service_end_date']}")
                            
                            if details_sections:
                                st.markdown("<p>" + "<br>".join(details_sections) + "</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p>No invoice details found.</p>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Payment info
                        payment_info = invoice["payment"]
                        
                        if any(payment_info.values()):
                            st.markdown(f"""
                            <div class="document-field">
                            <h4>💲 Payment Information</h4>
                            """, unsafe_allow_html=True)
                            
                            payment_sections = []
                            
                            if payment_info["currency"]:
                                payment_sections.append(f"<strong>Currency:</strong> {payment_info['currency']}")
                            
                            if payment_info["subtotal"]:
                                payment_sections.append(f"<strong>Subtotal:</strong> {payment_info['subtotal']}")
                            
                            if payment_info["total_tax"]:
                                payment_sections.append(f"<strong>Tax:</strong> {payment_info['total_tax']}")
                            
                            if payment_info["amount_due"]:
                                payment_sections.append(f"<strong>Amount Due:</strong> {payment_info['amount_due']}")
                            
                            if payment_info["previous_unpaid_balance"]:
                                payment_sections.append(f"<strong>Previous Balance:</strong> {payment_info['previous_unpaid_balance']}")
                            
                            if payment_sections:
                                st.markdown("<p>" + "<br>".join(payment_sections) + "</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p>No payment details found.</p>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.info("No invoice or payment details found.")
                    
                    with tabs[1]:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<h4>Vendor Information</h4>", unsafe_allow_html=True)
                            
                            # Vendor info
                            vendor_info = invoice["vendor"]
                            
                            if any(vendor_info.values()):
                                st.markdown(f"""
                                <div class="document-field">
                                <h4>🏢 {vendor_info["name"] or "Vendor"}</h4>
                                """, unsafe_allow_html=True)
                                
                                vendor_sections = []
                                
                                if vendor_info["address"]:
                                    vendor_sections.append(f"<strong>Address:</strong> {vendor_info['address']}")
                                
                                if vendor_info["tax_id"]:
                                    vendor_sections.append(f"<strong>Tax ID:</strong> {vendor_info['tax_id']}")
                                
                                if vendor_info["phone"]:
                                    vendor_sections.append(f"<strong>Phone:</strong> {vendor_info['phone']}")
                                
                                if vendor_info["email"]:
                                    vendor_sections.append(f"<strong>Email:</strong> {vendor_info['email']}")
                                
                                if vendor_info["website"]:
                                    vendor_sections.append(f"<strong>Website:</strong> {vendor_info['website']}")
                                
                                if vendor_sections:
                                    st.markdown("<p>" + "<br>".join(vendor_sections) + "</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("No vendor information found.")
                        
                        with col2:
                            st.markdown("<h4>Customer Information</h4>", unsafe_allow_html=True)
                            
                            # Customer info
                            customer_info = invoice["customer"]
                            
                            if any(customer_info.values()):
                                st.markdown(f"""
                                <div class="document-field">
                                <h4>👤 {customer_info["name"] or "Customer"}</h4>
                                """, unsafe_allow_html=True)
                                
                                customer_sections = []
                                
                                if customer_info["id"]:
                                    customer_sections.append(f"<strong>Customer ID:</strong> {customer_info['id']}")
                                
                                if customer_info["address"]:
                                    customer_sections.append(f"<strong>Address:</strong> {customer_info['address']}")
                                
                                if customer_info["shipping_address"]:
                                    customer_sections.append(f"<strong>Shipping Address:</strong> {customer_info['shipping_address']}")
                                
                                if customer_info["billing_address"]:
                                    customer_sections.append(f"<strong>Billing Address:</strong> {customer_info['billing_address']}")
                                
                                if customer_sections:
                                    st.markdown("<p>" + "<br>".join(customer_sections) + "</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("No customer information found.")
                    
                    with tabs[2]:
                        st.markdown("<h4>Line Items</h4>", unsafe_allow_html=True)
                        
                        if invoice["line_items"]:
                            # Create a table of items
                            items_data = []
                            
                            for item in invoice["line_items"]:
                                item_data = {
                                    "Description": item.get("description", "Unnamed Item"),
                                    "Quantity": item.get("quantity", ""),
                                    "Unit Price": item.get("unit_price", ""),
                                    "Amount": item.get("amount", ""),
                                    "Product Code": item.get("product_code", "")
                                }
                                
                                # Remove empty columns
                                item_data = {k: v for k, v in item_data.items() if v}
                                
                                items_data.append(item_data)
                            
                            # Display as a DataFrame
                            df = pd.DataFrame(items_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Calculate summary
                            item_count = len(invoice["line_items"])
                            
                            st.metric("Number of Line Items", item_count)
                        else:
                            st.info("No line items found on the invoice.")
                    
                    with tabs[3]:
                        st.subheader("Invoice Visualization")
                        
                        # Create calendar visualization for dates
                        dates = []
                        
                        if invoice["invoice_details"]["date"]:
                            try:
                                date_str = invoice["invoice_details"]["date"]
                                dates.append({"type": "Invoice Date", "date": date_str})
                            except:
                                pass
                        
                        if invoice["invoice_details"]["due_date"]:
                            try:
                                due_date_str = invoice["invoice_details"]["due_date"]
                                dates.append({"type": "Due Date", "date": due_date_str})
                            except:
                                pass
                        
                        if dates:
                            # Create a DataFrame
                            df = pd.DataFrame(dates)
                            
                            # Calculate payment window in days
                            payment_window = None
                            
                            if len(dates) == 2:
                                try:
                                    # Convert to datetime
                                    invoice_date = pd.to_datetime(df[df["type"] == "Invoice Date"]["date"].iloc[0])
                                    due_date = pd.to_datetime(df[df["type"] == "Due Date"]["date"].iloc[0])
                                    payment_window = (due_date - invoice_date).days
                                except:
                                    pass
                            
                            # Display date information
                            if payment_window is not None:
                                st.metric("Payment Window", f"{payment_window} days")
                        
                        # Payment breakdown
                        payment_data = {}
                        
                        if invoice["payment"]["subtotal"]:
                            try:
                                value_str = str(invoice["payment"]["subtotal"]).replace('$', '').replace(',', '')
                                payment_data["Subtotal"] = float(value_str)
                            except:
                                pass
                        
                        if invoice["payment"]["total_tax"]:
                            try:
                                value_str = str(invoice["payment"]["total_tax"]).replace('$', '').replace(',', '')
                                payment_data["Tax"] = float(value_str)
                            except:
                                pass
                        
                        # Calculate remaining amount (if any)
                        if "Subtotal" in payment_data and "Tax" in payment_data:
                            if invoice["payment"]["amount_due"]:
                                try:
                                    value_str = str(invoice["payment"]["amount_due"]).replace('$', '').replace(',', '')
                                    total = float(value_str)
                                    other = total - payment_data["Subtotal"] - payment_data["Tax"]
                                    if abs(other) > 0.01:  # Only add if significant
                                        payment_data["Other Charges"] = other
                                except:
                                    pass
                        
                        if payment_data:
                            # Create a DataFrame
                            df = pd.DataFrame({
                                "Category": list(payment_data.keys()),
                                "Amount": list(payment_data.values())
                            })
                            
                            # Create a pie chart
                            fig = px.pie(
                                df,
                                values="Amount",
                                names="Category",
                                title="Invoice Amount Breakdown",
                                color_discrete_sequence=px.colors.sequential.Viridis
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Line items analysis
                        if invoice["line_items"]:
                            items_with_amount = []
                            
                            for item in invoice["line_items"]:
                                if "description" in item and "amount" in item:
                                    try:
                                        if isinstance(item["amount"], str):
                                            # Remove currency symbols and commas
                                            cleaned_amount = item["amount"].replace('$', '').replace(',', '')
                                            amount = float(cleaned_amount)
                                        else:
                                            amount = float(item["amount"])
                                        
                                        description = item["description"]
                                        # Truncate long descriptions
                                        if len(description) > 30:
                                            description = description[:27] + "..."
                                        
                                        items_with_amount.append({
                                            "description": description,
                                            "amount": amount
                                        })
                                    except (ValueError, TypeError):
                                        pass
                            
                            if items_with_amount:
                                # Create a DataFrame
                                df = pd.DataFrame(items_with_amount)
                                
                                # Sort by amount
                                df = df.sort_values("amount", ascending=False)
                                
                                # Create a bar chart
                                fig = px.bar(
                                    df,
                                    x="description",
                                    y="amount",
                                    title="Line Item Amounts",
                                    color="amount",
                                    color_continuous_scale="viridis"
                                )
                                
                                fig.update_layout(xaxis_title="Item", yaxis_title="Amount")
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with tabs[4]:
                        st.subheader("Raw Extracted Data")
                        st.json(invoice)
                    
                    st.markdown("</div>", unsafe_allow_html=True)


def show_layout_page():
    """
    Display the document layout analysis page
    """
    st.header("Document Layout Analysis")
    
    st.markdown("""
    Extract document structure including text, tables, and layout information. Upload a document or use our sample.
    
    ### Key Capabilities
    
    - Identify document structure and layout
    - Extract text from documents with position information
    - Detect and parse tables with row/column relationships
    - Understand the reading order of text in complex layouts
    """)
    
    # Let user upload a file or use sample
    option = st.radio(
        "Choose an option",
        ["Upload my own document", "Use a sample document"]
    )
    
    document_path = None
    
    if option == "Upload my own document":
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "jpg", "jpeg", "png", "tiff"])
        if uploaded_file:
            document_path = save_uploaded_file(uploaded_file)
            try:
                # Try to display the document
                if document_path.lower().endswith('.pdf'):
                    # For PDFs, convert the first page to an image for display
                    image_path = convert_pdf_to_image(document_path)
                    if image_path:
                        st.image(image_path, caption="Uploaded Document (First Page)", use_container_width=True)
                    else:
                        st.warning("Unable to display PDF preview. Analysis will still work.")
                else:
                    st.image(document_path, caption="Uploaded Document", use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying document: {str(e)}")
    else:
        sample_path = SAMPLE_FILES["income_statement"]["path"]
        st.image(sample_path, caption="Sample Document", use_container_width=True)
        document_path = sample_path
    
    # Process the document
    if document_path and st.button("Analyze Document Layout"):
        with st.spinner("Analyzing document layout..."):
            start_time = time.time()
            result = analyze_document_layout(document_path=document_path)
            processing_time = time.time() - start_time
            
            if "error" in result:
                st.error(f"Error analyzing document: {result['error']}")
            else:
                st.success(f"Analysis completed in {processing_time:.2f} seconds!")
                
                # Create tabs for different aspects of the analysis
                tabs = st.tabs(["Document Overview", "Pages", "Tables", "Content", "Raw Data"])
                
                with tabs[0]:
                    st.markdown(f"""
                    <div class="document-card">
                    <h3>Document Overview</h3>
                    """, unsafe_allow_html=True)
                    
                    # Display summary info
                    has_handwritten = False
                    is_printed = False
                    if 'styles' in result and result['styles']:
                        for style in result['styles']:
                            if style.get('is_handwritten', False):
                                has_handwritten = True
                            else:
                                is_printed = True
                        
                        style_text = []
                        if has_handwritten:
                            style_text.append("Handwritten")
                        if is_printed:
                            style_text.append("Printed")
                        
                        if style_text:
                            st.markdown(f"**Document Style:** {', '.join(style_text)}")
                        else:
                            st.markdown("**Document Style:** Not detected")
                    else:
                        st.markdown("**Document Style:** Not detected")
                    st.markdown(f"**Pages:** {len(result['pages'])}")
                    st.markdown(f"**Tables:** {len(result['tables'])}")
                    st.markdown(f"**Paragraphs:** {len(result['paragraphs'])}")
                    
                    # Word count statistics
                    total_words = sum(len(page["words"]) for page in result["pages"])
                    total_lines = sum(len(page["lines"]) for page in result["pages"])
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Words", total_words)
                    col2.metric("Total Lines", total_lines)
                    
                    # Create a page word count chart
                    words_per_page = [len(page["words"]) for page in result["pages"]]
                    page_numbers = [f"Page {i+1}" for i in range(len(result["pages"]))]
                    
                    df = pd.DataFrame({
                        "Page": page_numbers,
                        "Words": words_per_page
                    })
                    
                    fig = px.bar(
                        df,
                        x="Page",
                        y="Words",
                        title="Word Count by Page",
                        color="Words",
                        color_continuous_scale="blues"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[1]:
                    st.markdown("<h3>Page Analysis</h3>", unsafe_allow_html=True)
                    
                    if len(result["pages"]) > 1:
                        page_index = st.selectbox(
                            "Select page to analyze",
                            range(len(result["pages"])),
                            format_func=lambda i: f"Page {i+1}"
                        )
                    else:
                        page_index = 0
                    
                    page = result["pages"][page_index]
                    
                    st.markdown(f"""
                    <div class="document-field">
                    <h4>Page {page_index + 1} Details</h4>
                    <p>Dimensions: {page["width"]} x {page["height"]} {page["unit"]}</p>
                    <p>Text Rotation: {page["text_angle"] if page["text_angle"] else "0"} degrees</p>
                    <p>Lines: {len(page["lines"])}</p>
                    <p>Words: {len(page["words"])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if page["lines"]:
                        st.markdown("<h4>Page Content</h4>", unsafe_allow_html=True)
                        page_content = "\n".join([line["text"] for line in page["lines"] if "text" in line])
                        st.text_area("", value=page_content, height=300)
                    else:
                        st.info("No page content available.")
                    
                    if page["lines"]:
                        st.markdown("<h4>Top Lines by Length</h4>", unsafe_allow_html=True)
                        line_lengths = [len(line["text"]) for line in page["lines"]]
                        line_numbers = [f"Line {i+1}" for i in range(len(page["lines"]))]
                        
                        lines_df = pd.DataFrame({
                            "Line Number": line_numbers,
                            "Text": [line["text"] for line in page["lines"]],
                            "Length": line_lengths
                        })
                        
                        lines_df = lines_df.sort_values("Length", ascending=False).head(10)
                        st.dataframe(lines_df, use_container_width=True)
                
                with tabs[2]:
                    st.markdown("<h3>Table Analysis</h3>", unsafe_allow_html=True)
                    
                    if result["tables"]:
                        def get_table_page_number(table):
                            """Extract page number from a table's bounding regions if available."""
                            if "bounding_regions" in table and table["bounding_regions"]:
                                return f"Page {table['bounding_regions'][0]['page_number']}"
                            return "Unknown page"
                        
                        table_index = st.selectbox(
                            "Select table to view",
                            range(len(result["tables"])),
                            format_func=lambda i: f"Table {i+1} ({get_table_page_number(result['tables'][i])})"
                        )
                        
                        table = result["tables"][table_index]
                        
                        st.markdown(f"""
                        <div class="document-field">
                        <h4>Table {table_index + 1} Details</h4>
                        <p>Dimensions: {table["row_count"]} rows x {table["column_count"]} columns</p>
                        <p>Page: {get_table_page_number(table)}</p>
                        <p>Cells: {len(table["cells"])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<h4>Table Content</h4>", unsafe_allow_html=True)
                        
                        table_data = {}
                        for cell in table["cells"]:
                            row = cell["row_index"]
                            col = cell["column_index"]
                            text = cell["text"]
                            if row not in table_data:
                                table_data[row] = {}
                            table_data[row][col] = text
                        
                        table_rows = []
                        for row_idx in range(table["row_count"]):
                            if row_idx in table_data:
                                row_cells = []
                                for col_idx in range(table["column_count"]):
                                    cell_text = table_data[row_idx].get(col_idx, "")
                                    row_cells.append(cell_text)
                                table_rows.append(row_cells)
                            else:
                                table_rows.append([""] * table["column_count"])
                        
                        col_names = [chr(65 + i) for i in range(table["column_count"])]
                        df = pd.DataFrame(table_rows, columns=col_names)
                        st.dataframe(df, use_container_width=True)
                        
                        st.markdown("<h4>Cell Content Analysis</h4>", unsafe_allow_html=True)
                        cell_lengths = [len(cell["text"]) for cell in table["cells"]]
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Average Cell Length", f"{sum(cell_lengths) / len(cell_lengths):.1f}" if cell_lengths else "N/A")
                        col2.metric("Max Cell Length", f"{max(cell_lengths)}" if cell_lengths else "N/A")
                        col3.metric("Min Cell Length", f"{min(cell_lengths)}" if cell_lengths else "N/A")
                        
                        fig = px.histogram(
                            x=cell_lengths,
                            title="Cell Length Distribution",
                            labels={"x": "Cell Length", "y": "Count"},
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No tables found in the document.")
                
                with tabs[3]:
                    st.markdown("<h3>Document Content</h3>", unsafe_allow_html=True)
                    
                    if "content" in result and result["content"]:
                        st.markdown("<h4>Full Document Text</h4>", unsafe_allow_html=True)
                        st.text_area("", value=result["content"], height=400)
                    
                    if result["paragraphs"]:
                        st.markdown("<h4>Paragraphs</h4>", unsafe_allow_html=True)
                        for i, para in enumerate(result["paragraphs"][:10]):
                            role = para.get("role", "")
                            st.markdown(f"""
                            <div class="document-field">
                            <p><strong>Paragraph {i+1}{' - ' + role if role else ''}</strong></p>
                            <p style="white-space: pre-wrap;">{para["text"]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(result["paragraphs"]) > 10:
                            st.info(f"Showing 10 of {len(result['paragraphs'])} paragraphs. See Raw Data for all.")
                
                with tabs[4]:
                    st.subheader("Raw Extracted Data")
                    
                    simplified_result = {
                        "pages": [
                            {
                                "page_number": page["page_number"],
                                "width": page["width"],
                                "height": page["height"],
                                "unit": page["unit"],
                                "text_angle": page["text_angle"],
                                "line_count": len(page["lines"]),
                                "word_count": len(page["words"])
                            }
                            for page in result["pages"]
                        ],
                        "tables": [
                            {
                                "table_number": table["table_number"],
                                "row_count": table["row_count"],
                                "column_count": table["column_count"],
                                "page_number": table["bounding_regions"][0]["page_number"] if table["bounding_regions"] else "Unknown",
                                "cell_count": len(table["cells"])
                            }
                            for table in result["tables"]
                        ],
                        "paragraph_count": len(result["paragraphs"]),
                        "content_length": len(result["content"]) if "content" in result else 0,
                        "word_count": result.get("word_count", 0),
                        "character_count": result.get("character_count", 0)
                    }
                    
                    st.json(simplified_result)

def show_general_document_page():
    """
    Display the general document analysis page
    """
    st.header("General Document Analysis")
    
    st.markdown("""
    Analyze any document type using Azure Document Intelligence. Upload a document or use our sample.
    
    ### Key Capabilities
    
    - Extract text from any document (OCR)
    - Identify key-value pairs in documents
    - Detect document entities and relationships
    - Understand document structure and content
    """)
    
    # Let user upload a file or use sample
    option = st.radio(
        "Choose an option",
        ["Upload my own document", "Use a sample document"]
    )
    
    document_path = None
    
    if option == "Upload my own document":
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "jpg", "jpeg", "png", "tiff"])
        if uploaded_file:
            document_path = save_uploaded_file(uploaded_file)
            try:
                # Try to display the document
                if document_path.lower().endswith('.pdf'):
                    # For PDFs, convert the first page to an image for display
                    image_path = convert_pdf_to_image(document_path)
                    if image_path:
                        st.image(image_path, caption="Uploaded Document (First Page)", use_container_width=True)
                    else:
                        st.warning("Unable to display PDF preview. Analysis will still work.")
                else:
                    st.image(document_path, caption="Uploaded Document", use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying document: {str(e)}")
    else:
        sample_path = SAMPLE_FILES["layout"]["path"]
        # For PDFs, convert the first page to an image for display
        if sample_path.lower().endswith('.pdf'):
            image_path = convert_pdf_to_image(sample_path)
            if image_path:
                st.image(image_path, caption="Sample Document (First Page)", use_container_width=True)
        else:
            st.image(sample_path, caption="Sample Document", use_container_width=True)
        document_path = sample_path
    
    # Analysis options
    st.subheader("Analysis Options")
    
    analysis_options = st.multiselect(
        "Select analysis types",
        ["Text Extraction (OCR)", "Document Analysis", "Key-Value Pair Extraction"],
        default=["Text Extraction (OCR)", "Key-Value Pair Extraction"]
    )
    
    # Process the document
    if document_path and analysis_options and st.button("Analyze Document"):
        with st.spinner("Analyzing document..."):
            results = {}
            
            # Perform selected analyses
            if "Text Extraction (OCR)" in analysis_options:
                start_time = time.time()
                text_result = extract_text(document_path=document_path)
                text_time = time.time() - start_time
                
                if "error" not in text_result:
                    results["text"] = {
                        "result": text_result,
                        "time": text_time
                    }
                else:
                    st.error(f"Error in text extraction: {text_result['error']}")
            
            if "Document Analysis" in analysis_options or "Key-Value Pair Extraction" in analysis_options:
                start_time = time.time()
                doc_result = analyze_document(document_path=document_path)
                doc_time = time.time() - start_time
                
                if "error" not in doc_result:
                    results["document"] = {
                        "result": doc_result,
                        "time": doc_time
                    }
                else:
                    st.error(f"Error in document analysis: {doc_result['error']}")
            
            # Display results
            if results:
                st.success(f"Analysis completed successfully!")
                
                # Create tabs for different analyses
                tabs = []
                
                if "text" in results:
                    tabs.append("Text Extraction")
                
                if "document" in results:
                    if "Key-Value Pair Extraction" in analysis_options:
                        tabs.append("Key-Value Pairs")
                    
                    if "Document Analysis" in analysis_options:
                        tabs.append("Document Analysis")
                
                tabs.append("Raw Data")
                
                # Create tab views
                tab_views = st.tabs(tabs)
                tab_index = 0
                
                # Text extraction tab
                if "text" in results and tab_index < len(tab_views):
                    with tab_views[tab_index]:
                        text_result = results["text"]["result"]
                        
                        st.markdown(f"""
                        <div class="document-card">
                        <h3>Text Extraction Results</h3>
                        <p>Processing Time: {results['text']['time']:.2f} seconds</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display document details
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Pages", len(text_result["pages"]))
                        col2.metric("Words", text_result["word_count"])
                        col3.metric("Characters", text_result["character_count"])
                        
                        # Display extracted text
                        st.markdown("<h4>Extracted Text</h4>", unsafe_allow_html=True)
                        st.text_area("", value=text_result["content"], height=300)
                        
                        # Display languages (updated)
                        if text_result["languages"]:
                            languages_str = ", ".join([f"{lang['language_code']} ({lang['confidence']:.2f})" for lang in text_result["languages"]])
                            st.info(f"Detected languages: {languages_str}")
                        else:
                            st.info("No languages detected.")
                        
                        # (Rest of the tab code remains unchanged)
                    
                    tab_index += 1
                
                # Key-value pairs tab
                if "document" in results and "Key-Value Pair Extraction" in analysis_options and tab_index < len(tab_views):
                    with tab_views[tab_index]:
                        doc_result = results["document"]["result"]
                        
                        st.markdown(f"""
                        <div class="document-card">
                        <h3>Key-Value Pair Extraction</h3>
                        <p>Processing Time: {results['document']['time']:.2f} seconds</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display key-value pairs
                        if doc_result["key_value_pairs"]:
                            # Display in a nice layout
                            for kv_pair in doc_result["key_value_pairs"]:
                                confidence_color = "green" if kv_pair["confidence"] > 0.8 else "orange" if kv_pair["confidence"] > 0.6 else "red"
                                
                                st.markdown(f"""
                                <div class="document-field" style="position: relative;">
                                <div style="position: absolute; top: 10px; right: 10px; background-color: {confidence_color}; color: white; padding: 2px 5px; border-radius: 3px; font-size: 0.8em;">
                                {kv_pair["confidence"]:.2%}
                                </div>
                                <p><strong>{kv_pair["key"]}</strong></p>
                                <p>{kv_pair["value"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Create a chart of confidence scores
                            if len(doc_result["key_value_pairs"]) > 1:
                                # Create DataFrame
                                df = pd.DataFrame({
                                    "Key": [kv["key"] for kv in doc_result["key_value_pairs"]],
                                    "Confidence": [kv["confidence"] for kv in doc_result["key_value_pairs"]]
                                })
                                
                                # Sort by confidence
                                df = df.sort_values("Confidence")
                                
                                # Truncate long keys
                                df["Key"] = df["Key"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)
                                
                                # Create chart
                                fig = px.bar(
                                    df,
                                    x="Key",
                                    y="Confidence",
                                    title="Key-Value Pair Confidence",
                                    color="Confidence",
                                    color_continuous_scale="RdYlGn",
                                    range_color=[0, 1]
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No key-value pairs found in the document.")
                    
                    tab_index += 1
                
                # Document analysis tab
                if "document" in results and "Document Analysis" in analysis_options and tab_index < len(tab_views):
                    with tab_views[tab_index]:
                        doc_result = results["document"]["result"]
                        
                        st.markdown(f"""
                        <div class="document-card">
                        <h3>Document Analysis Results</h3>
                        <p>Processing Time: {results['document']['time']:.2f} seconds</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display document type if available
                        if doc_result["document_type"]:
                            st.info(f"Document Type: {doc_result['document_type']}")
                        
                        # Display entities
                        if doc_result["entities"]:
                            st.markdown("<h4>Detected Entities</h4>", unsafe_allow_html=True)
                            
                            # Create DataFrame
                            df = pd.DataFrame({
                                "Category": [entity["category"] for entity in doc_result["entities"]],
                                "Subcategory": [entity["subcategory"] if entity["subcategory"] else "" for entity in doc_result["entities"]],
                                "Content": [entity["content"] for entity in doc_result["entities"]],
                                "Confidence": [entity["confidence"] for entity in doc_result["entities"]]
                            })
                            
                            # Display as a table
                            st.dataframe(df, use_container_width=True)
                            
                            # Count entities by category
                            category_counts = df["Category"].value_counts().reset_index()
                            category_counts.columns = ["Category", "Count"]
                            
                            # Create a pie chart
                            fig = px.pie(
                                category_counts,
                                values="Count",
                                names="Category",
                                title="Entity Categories"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display tables
                        if doc_result["tables"]:
                            st.markdown("<h4>Detected Tables</h4>", unsafe_allow_html=True)
                            
                            # Table selector
                            table_index = st.selectbox(
                                "Select table to view",
                                range(len(doc_result["tables"])),
                                format_func=lambda i: f"Table {i+1} ({doc_result['tables'][i]['row_count']} rows × {doc_result['tables'][i]['column_count']} columns)"
                            )
                            
                            # Get selected table
                            table = doc_result["tables"][table_index]
                            
                            # Reconstruct and display the table
                            table_data = {}
                            for cell in table["cells"]:
                                row = cell["row_index"]
                                col = cell["column_index"]
                                text = cell["text"]
                                
                                if row not in table_data:
                                    table_data[row] = {}
                                
                                table_data[row][col] = text
                            
                            # Convert to DataFrame
                            table_rows = []
                            for row_idx in range(table["row_count"]):
                                if row_idx in table_data:
                                    row_cells = []
                                    for col_idx in range(table["column_count"]):
                                        cell_text = table_data[row_idx].get(col_idx, "")
                                        row_cells.append(cell_text)
                                    table_rows.append(row_cells)
                                else:
                                    table_rows.append([""] * table["column_count"])
                            
                            # Create column names (A, B, C, etc.)
                            col_names = [chr(65 + i) for i in range(table["column_count"])]
                            
                            # Create DataFrame
                            df = pd.DataFrame(table_rows, columns=col_names)
                            
                            # Display table
                            st.dataframe(df, use_container_width=True)
                    
                    tab_index += 1
                
                # Raw data tab
                if tab_index < len(tab_views):
                    with tab_views[tab_index]:
                        st.subheader("Raw Extracted Data")
                        
                        if "text" in results:
                            with st.expander("Text Extraction Raw Data"):
                                simplified_text_result = {
                                    "languages": results["text"]["result"]["languages"],  # Updated from "language"
                                    "pages": len(results["text"]["result"]["pages"]),
                                    "word_count": results["text"]["result"]["word_count"],
                                    "character_count": results["text"]["result"]["character_count"],
                                    "content_length": len(results["text"]["result"]["content"])
                                }
                                st.json(simplified_text_result)
                        
                        if "document" in results:
                            with st.expander("Document Analysis Raw Data"):
                                st.json(results["document"]["result"])




