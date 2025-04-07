"""
Receipt Analysis with Azure Document Intelligence
"""

import os
from datetime import datetime
from .client import get_document_intelligence_client

def analyze_receipt(image_path=None, image_url=None):
    """
    Analyze a receipt image using Document Intelligence
    
    Args:
        image_path (str, optional): Path to a local receipt image
        image_url (str, optional): URL of a receipt image
        
    Returns:
        dict: Structured data extracted from the receipt
    """
    client = get_document_intelligence_client()
    if not client:
        return None
    
    try:
        # Process the receipt
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as image:
                poller = client.begin_analyze_document(
                    "prebuilt-receipt", 
                    image
                )
        elif image_url:
            poller = client.begin_analyze_document(
                "prebuilt-receipt", 
                image_url
            )
        else:
            return {"error": "No valid image path or URL provided"}
        
        result = poller.result()
        
        if not result.documents or len(result.documents) == 0:
            return {"error": "No receipt found in the image"}
        
        # Extract receipt data
        extracted_data = []
        
        for receipt_idx, receipt in enumerate(result.documents):
            receipt_data = {
                "receipt_index": receipt_idx + 1,
                "confidence": receipt.confidence,
                "merchant": None,
                "transaction": {
                    "date": None,
                    "time": None,
                    "total": None,
                    "subtotal": None,
                    "tax": None,
                    "tip": None
                },
                "items": [],
                "payment_info": {
                    "card_type": None,
                    "card_number": None
                },
                "contact_info": {
                    "phone": None,
                    "address": None,
                    "merchant_url": None
                }
            }
            
            # Extract fields from receipt
            for field_name, field in receipt.fields.items():
                
                # Merchant information
                if field_name == "MerchantName" and field.content:
                    receipt_data["merchant"] = field.content
                
                # Transaction details
                elif field_name == "TransactionDate" and field.content:
                    # Format the date if it's a datetime object
                    if isinstance(field.content, datetime):
                        receipt_data["transaction"]["date"] = field.content.strftime("%Y-%m-%d")
                    else:
                        receipt_data["transaction"]["date"] = field.content
                
                elif field_name == "TransactionTime" and field.content:
                    # Format the time if it's a datetime object
                    if isinstance(field.content, datetime):
                        receipt_data["transaction"]["time"] = field.content.strftime("%H:%M:%S")
                    else:
                        receipt_data["transaction"]["time"] = field.content
                
                elif field_name == "Total" and field.content:
                    receipt_data["transaction"]["total"] = field.content
                
                elif field_name == "Subtotal" and field.content:
                    receipt_data["transaction"]["subtotal"] = field.content
                
                elif field_name == "TotalTax" and field.content:
                    receipt_data["transaction"]["tax"] = field.content
                
                elif field_name == "Tip" and field.content:
                    receipt_data["transaction"]["tip"] = field.content
                
                # Line items
                elif field_name == "Items" and hasattr(field, "value_array"):
                    for item in field.value_array:
                        item_data = {}
                        
                        # Item should have valueObject property
                        if hasattr(item, "value_object"):
                            item_obj = item.value_object
                            
                            # Extract item details - Description instead of Name
                            if "Description" in item_obj and hasattr(item_obj["Description"], "value_string"):
                                item_data["name"] = item_obj["Description"].value_string
                            
                            # Quantity
                            if "Quantity" in item_obj and hasattr(item_obj["Quantity"], "value_number"):
                                item_data["quantity"] = item_obj["Quantity"].value_number
                            
                            # Price (might not always be present)
                            if "Price" in item_obj and hasattr(item_obj["Price"], "value_currency"):
                                item_data["price"] = item_obj["Price"].value_currency.amount
                            
                            # TotalPrice
                            if "TotalPrice" in item_obj and hasattr(item_obj["TotalPrice"], "value_currency"):
                                price_obj = item_obj["TotalPrice"].value_currency
                                item_data["total_price"] = price_obj.amount
                                
                                # Add currency code if available
                                if hasattr(price_obj, "currency_code") and price_obj.currency_code:
                                    item_data["currency"] = price_obj.currency_code
                        
                        if item_data:
                            receipt_data["items"].append(item_data)
                
                # Payment information
                elif field_name == "PaymentType" and field.content:
                    receipt_data["payment_info"]["card_type"] = field.content
                
                elif field_name == "PaymentCardNumber" and field.content:
                    receipt_data["payment_info"]["card_number"] = field.content
                
                # Contact information
                elif field_name == "MerchantPhoneNumber" and field.content:
                    receipt_data["contact_info"]["phone"] = field.content
                
                elif field_name == "MerchantAddress" and field.content:
                    receipt_data["contact_info"]["address"] = field.content
                
                elif field_name == "MerchantUrl" and field.content:
                    receipt_data["contact_info"]["merchant_url"] = field.content
            
            extracted_data.append(receipt_data)
        
        return {"receipts": extracted_data}
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Sample receipt URL
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/contoso-receipt.png"
    
    # Analyze from URL
    result = analyze_receipt(image_url=sample_url)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for receipt in result["receipts"]:
            print(f"\n----- Receipt #{receipt['receipt_index']} -----")
            print(f"Confidence: {receipt['confidence']:.4f}")
            
            if receipt["merchant"]:
                print(f"\nMerchant: {receipt['merchant']}")
            
            print("\nTransaction Details:")
            for detail, value in receipt["transaction"].items():
                if value is not None:
                    print(f"  - {detail.replace('_', ' ').title()}: {value}")
            
            if receipt["items"]:
                print("\nItems:")
                for idx, item in enumerate(receipt["items"]):
                    print(f"  {idx + 1}. {item.get('name', 'Unknown Item')}")
                    for key, value in item.items():
                        if key != "name":
                            print(f"     - {key.replace('_', ' ').title()}: {value}")
            
            for section_name, section_data in [
                ("Payment Information", receipt["payment_info"]),
                ("Contact Information", receipt["contact_info"])
            ]:
                has_data = any(value is not None for value in section_data.values())
                if has_data:
                    print(f"\n{section_name}:")
                    for key, value in section_data.items():
                        if value is not None:
                            print(f"  - {key.replace('_', ' ').title()}: {value}")