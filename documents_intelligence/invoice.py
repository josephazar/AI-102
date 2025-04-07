"""
Invoice Analysis with Azure Document Intelligence
"""

import os
from datetime import datetime
from .client import get_document_intelligence_client

def analyze_invoice(document_path=None, document_url=None):
    """
    Analyze an invoice document using Document Intelligence
    
    Args:
        document_path (str, optional): Path to a local invoice document
        document_url (str, optional): URL of an invoice document
        
    Returns:
        dict: Structured data extracted from the invoice
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the invoice
        if document_path and os.path.isfile(document_path):
            with open(document_path, "rb") as document:
                poller = client.begin_analyze_document(
                    "prebuilt-invoice", 
                    document, 
                )
        elif document_url:
            poller = client.begin_analyze_document(
                "prebuilt-invoice", 
                document_url
            )
        else:
            return {"error": "No valid invoice path or URL provided"}
        
        result = poller.result()
        
        if not result.documents or len(result.documents) == 0:
            return {"error": "No invoice found in the document"}
        
        # Extract invoice data
        extracted_data = []
        
        for invoice_idx, invoice in enumerate(result.documents):
            invoice_data = {
                "invoice_index": invoice_idx + 1,
                "confidence": invoice.confidence,
                "vendor": {
                    "name": None,
                    "address": None,
                    "phone": None,
                    "tax_id": None,
                    "email": None,
                    "website": None
                },
                "customer": {
                    "name": None,
                    "id": None,
                    "address": None,
                    "shipping_address": None,
                    "billing_address": None
                },
                "invoice_details": {
                    "id": None,
                    "date": None,
                    "due_date": None,
                    "purchase_order": None,
                    "service_start_date": None,
                    "service_end_date": None
                },
                "payment": {
                    "currency": None,
                    "subtotal": None,
                    "total_tax": None,
                    "previous_unpaid_balance": None,
                    "amount_due": None
                },
                "line_items": []
            }
            
            # Extract fields from invoice
            for field_name, field in invoice.fields.items():
                # Vendor information
                if field_name == "VendorName" and field.content:
                    invoice_data["vendor"]["name"] = field.content
                elif field_name == "VendorAddress" and field.content:
                    invoice_data["vendor"]["address"] = field.content
                elif field_name == "VendorAddressRecipient" and field.content:
                    invoice_data["vendor"]["name"] = field.content
                elif field_name == "Phone" and field.content:
                    invoice_data["vendor"]["phone"] = field.content
                elif field_name == "VendorTaxId" and field.content:
                    invoice_data["vendor"]["tax_id"] = field.content
                elif field_name == "Email" and field.content:
                    invoice_data["vendor"]["email"] = field.content
                elif field_name == "Website" and field.content:
                    invoice_data["vendor"]["website"] = field.content
                
                # Customer information
                elif field_name == "CustomerName" and field.content:
                    invoice_data["customer"]["name"] = field.content
                elif field_name == "CustomerId" and field.content:
                    invoice_data["customer"]["id"] = field.content
                elif field_name == "CustomerAddress" and field.content:
                    invoice_data["customer"]["address"] = field.content
                elif field_name == "CustomerAddressRecipient" and field.content:
                    invoice_data["customer"]["name"] = invoice_data["customer"]["name"] or field.content
                elif field_name == "ShippingAddress" and field.content:
                    invoice_data["customer"]["shipping_address"] = field.content
                elif field_name == "BillingAddress" and field.content:
                    invoice_data["customer"]["billing_address"] = field.content
                
                # Invoice details
                elif field_name == "InvoiceId" and field.content:
                    invoice_data["invoice_details"]["id"] = field.content
                elif field_name == "InvoiceDate" and field.content:
                    # Format the date if it's a datetime object
                    if isinstance(field.content, datetime):
                        invoice_data["invoice_details"]["date"] = field.content.strftime("%Y-%m-%d")
                    else:
                        invoice_data["invoice_details"]["date"] = field.content
                elif field_name == "DueDate" and field.content:
                    # Format the date if it's a datetime object
                    if isinstance(field.content, datetime):
                        invoice_data["invoice_details"]["due_date"] = field.content.strftime("%Y-%m-%d")
                    else:
                        invoice_data["invoice_details"]["due_date"] = field.content
                elif field_name == "PurchaseOrder" and field.content:
                    invoice_data["invoice_details"]["purchase_order"] = field.content
                elif field_name == "ServiceStartDate" and field.content:
                    # Format the date if it's a datetime object
                    if isinstance(field.content, datetime):
                        invoice_data["invoice_details"]["service_start_date"] = field.content.strftime("%Y-%m-%d")
                    else:
                        invoice_data["invoice_details"]["service_start_date"] = field.content
                elif field_name == "ServiceEndDate" and field.content:
                    # Format the date if it's a datetime object
                    if isinstance(field.content, datetime):
                        invoice_data["invoice_details"]["service_end_date"] = field.content.strftime("%Y-%m-%d")
                    else:
                        invoice_data["invoice_details"]["service_end_date"] = field.content
                
                # Payment information
                elif field_name == "InvoiceTotal" and field.content:
                    invoice_data["payment"]["amount_due"] = field.content
                    # Try to extract currency
                    if hasattr(field, "currencies") and field.currencies and len(field.currencies) > 0:
                        invoice_data["payment"]["currency"] = field.currencies[0]
                elif field_name == "SubTotal" and field.content:
                    invoice_data["payment"]["subtotal"] = field.content
                elif field_name == "TotalTax" and field.content:
                    invoice_data["payment"]["total_tax"] = field.content
                elif field_name == "PreviousUnpaidBalance" and field.content:
                    invoice_data["payment"]["previous_unpaid_balance"] = field.content
                elif field_name == "AmountDue" and field.content:
                    invoice_data["payment"]["amount_due"] = field.content
                elif field_name == "PaymentTerm" and field.content:
                    invoice_data["payment"]["payment_term"] = field.content
                
                # Line items
                elif field_name == "Items" and hasattr(field, "value_array"):
                    for item in field.value_array:
                        item_data = {}
                        
                        # Item should have valueObject property
                        if hasattr(item, "value_object"):
                            item_obj = item.value_object
                            
                            # Extract item details
                            if "Description" in item_obj and hasattr(item_obj["Description"], "value_string"):
                                item_data["description"] = item_obj["Description"].value_string
                            
                            # Quantity
                            if "Quantity" in item_obj and hasattr(item_obj["Quantity"], "value_number"):
                                item_data["quantity"] = item_obj["Quantity"].value_number
                            
                            # UnitPrice
                            if "UnitPrice" in item_obj and hasattr(item_obj["UnitPrice"], "value_currency"):
                                unit_price_obj = item_obj["UnitPrice"].value_currency
                                item_data["unit_price"] = unit_price_obj.amount
                                
                                # Add currency code if available
                                if hasattr(unit_price_obj, "currency_code") and unit_price_obj.currency_code:
                                    item_data["currency"] = unit_price_obj.currency_code
                            
                            # Amount
                            if "Amount" in item_obj and hasattr(item_obj["Amount"], "value_currency"):
                                amount_obj = item_obj["Amount"].value_currency
                                item_data["amount"] = amount_obj.amount
                                
                                # Add currency code if available and not already set
                                if "currency" not in item_data and hasattr(amount_obj, "currency_code") and amount_obj.currency_code:
                                    item_data["currency"] = amount_obj.currency_code
                            
                            # ProductCode
                            if "ProductCode" in item_obj and hasattr(item_obj["ProductCode"], "value_string"):
                                item_data["product_code"] = item_obj["ProductCode"].value_string
                            
                            # Date
                            if "Date" in item_obj and hasattr(item_obj["Date"], "value_date"):
                                date_value = item_obj["Date"].value_date
                                # Format the date if it's a datetime object
                                if isinstance(date_value, datetime):
                                    item_data["date"] = date_value.strftime("%Y-%m-%d")
                                else:
                                    item_data["date"] = str(date_value)
                        
                        if item_data:
                            invoice_data["line_items"].append(item_data)
            
            extracted_data.append(invoice_data)
        
        return {"invoices": extracted_data}
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Sample invoice URL
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf"
    
    # Analyze from URL
    result = analyze_invoice(document_url=sample_url)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for invoice in result["invoices"]:
            print(f"\n----- Invoice #{invoice['invoice_index']} -----")
            print(f"Confidence: {invoice['confidence']:.4f}")
            
            # Vendor information
            if any(value is not None for value in invoice["vendor"].values()):
                print("\nVendor Information:")
                for key, value in invoice["vendor"].items():
                    if value:
                        print(f"  - {key.replace('_', ' ').title()}: {value}")
            
            # Customer information
            if any(value is not None for value in invoice["customer"].values()):
                print("\nCustomer Information:")
                for key, value in invoice["customer"].items():
                    if value:
                        print(f"  - {key.replace('_', ' ').title()}: {value}")
            
            # Invoice details
            if any(value is not None for value in invoice["invoice_details"].values()):
                print("\nInvoice Details:")
                for key, value in invoice["invoice_details"].items():
                    if value:
                        print(f"  - {key.replace('_', ' ').title()}: {value}")
            
            # Payment information
            if any(value is not None for value in invoice["payment"].values()):
                print("\nPayment Information:")
                for key, value in invoice["payment"].items():
                    if value:
                        print(f"  - {key.replace('_', ' ').title()}: {value}")
            
            # Line items
            if invoice["line_items"]:
                print("\nLine Items:")
                for idx, item in enumerate(invoice["line_items"]):
                    print(f"  {idx + 1}. {item.get('description', 'Unnamed Item')}")
                    for key, value in item.items():
                        if key != "description" and value:
                            print(f"     - {key.replace('_', ' ').title()}: {value}")