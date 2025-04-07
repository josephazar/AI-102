"""
Business Card Analysis with Azure Document Intelligence
"""

import os
from .client import get_document_intelligence_client

def analyze_business_card(image_path=None, image_url=None):
    """
    Analyze business card image using the Document Intelligence API and extract relevant information.
    
    Args:
        image_path (str, optional): Local path to the business card image
        image_url (str, optional): URL to the business card image
    
    Returns:
        dict: Dictionary containing extracted business card information
    """
    try:
        client = get_document_intelligence_client()
        
        if not client:
            return {"error": "Document Intelligence client not initialized"}
        
        poller = None
        
        # The model ID has changed in newer versions of the SDK
        # Use "prebuilt-businessCard" or one of the supported model IDs
        model_id = "prebuilt-businessCard"
        
        if image_path:
            # Check if file exists
            if not os.path.isfile(image_path):
                return {"error": f"File not found: {image_path}"}
            
            # Open the file and pass it directly
            with open(image_path, "rb") as file:
                poller = client.begin_analyze_document(
                    model_id,
                    file
                )
                
        elif image_url:
            # Use the correct method for URL-based analysis
            poller = client.begin_analyze_document(
                model_id,
                image_url
            )
            
        else:
            return {"error": "Either image_path or image_url must be provided"}
        
        # Get results
        result = poller.result()
        
        # Process results
        cards = []
        
        for document in result.documents:
            card_data = {
                "confidence": document.confidence,
                "contacts": [],
                "company": [],
                "departments": [],
                "job_titles": [],
                "emails": [],
                "websites": [],
                "addresses": [],
                "phone_numbers": {
                    "mobile": [],
                    "work": [],
                    "fax": [],
                    "other": []
                }
            }
            
            # Extract fields
            for field_name, field in document.fields.items():
                # Handle the ContactNames field
                if field_name == "ContactNames" and field.value:
                    for contact in field.value:
                        contact_data = {}
                        if "FirstName" in contact.value and contact.value["FirstName"].value:
                            contact_data["first_name"] = contact.value["FirstName"].value
                        if "LastName" in contact.value and contact.value["LastName"].value:
                            contact_data["last_name"] = contact.value["LastName"].value
                        if contact_data:
                            card_data["contacts"].append(contact_data)
                
                # Handle the CompanyNames field
                elif field_name == "CompanyNames" and field.value:
                    for company in field.value:
                        if company.value:
                            card_data["company"].append(company.value)
                
                # Handle the Departments field
                elif field_name == "Departments" and field.value:
                    for dept in field.value:
                        if dept.value:
                            card_data["departments"].append(dept.value)
                
                # Handle the JobTitles field
                elif field_name == "JobTitles" and field.value:
                    for title in field.value:
                        if title.value:
                            card_data["job_titles"].append(title.value)
                
                # Handle the Emails field
                elif field_name == "Emails" and field.value:
                    for email in field.value:
                        if email.value:
                            card_data["emails"].append(email.value)
                
                # Handle the Websites field
                elif field_name == "Websites" and field.value:
                    for website in field.value:
                        if website.value:
                            card_data["websites"].append(website.value)
                
                # Handle the Addresses field
                elif field_name == "Addresses" and field.value:
                    for address in field.value:
                        if address.value:
                            card_data["addresses"].append(address.value)
                
                # Handle the MobilePhones field
                elif field_name == "MobilePhones" and field.value:
                    for phone in field.value:
                        if phone.value:
                            card_data["phone_numbers"]["mobile"].append(phone.value)
                
                # Handle the WorkPhones field
                elif field_name == "WorkPhones" and field.value:
                    for phone in field.value:
                        if phone.value:
                            card_data["phone_numbers"]["work"].append(phone.value)
                
                # Handle the Faxes field
                elif field_name == "Faxes" and field.value:
                    for fax in field.value:
                        if fax.value:
                            card_data["phone_numbers"]["fax"].append(fax.value)
                
                # Handle the OtherPhones field
                elif field_name == "OtherPhones" and field.value:
                    for phone in field.value:
                        if phone.value:
                            card_data["phone_numbers"]["other"].append(phone.value)
            
            cards.append(card_data)
        
        return {"cards": cards}
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    # Sample business card URL
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/java/ComputerVision/BusinessCard.jpg"
    
    # Analyze from URL
    result = analyze_business_card(image_url=sample_url)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for i, card in enumerate(result["cards"]):
            print(f"\n----- Business Card #{i+1} -----")
            print(f"Confidence: {card['confidence']:.4f}")
            
            if card["contacts"]:
                print("\nContacts:")
                for contact in card["contacts"]:
                    name_parts = []
                    if "first_name" in contact:
                        name_parts.append(contact["first_name"])
                    if "last_name" in contact:
                        name_parts.append(contact["last_name"])
                    
                    print(f"  - {' '.join(name_parts)}")
            
            if card["company"]:
                print("\nCompany:")
                for company in card["company"]:
                    print(f"  - {company}")
            
            if card["job_titles"]:
                print("\nJob Titles:")
                for title in card["job_titles"]:
                    print(f"  - {title}")
            
            if card["emails"]:
                print("\nEmails:")
                for email in card["emails"]:
                    print(f"  - {email}")
            
            if card["websites"]:
                print("\nWebsites:")
                for website in card["websites"]:
                    print(f"  - {website}")
            
            if card["addresses"]:
                print("\nAddresses:")
                for address in card["addresses"]:
                    print(f"  - {address}")
            
            for phone_type, phones in card["phone_numbers"].items():
                if phones:
                    print(f"\n{phone_type.title()} Phones:")
                    for phone in phones:
                        print(f"  - {phone}")