import yaml
from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential

# get the information about the connection
with open("../account_configs.yml", "r") as f:
    data = yaml.load(f, yaml.FullLoader)
    subscription_key = data["fr_subscription_key"]
    endpoint = data["fr_endpoint"]

# pre-trained client connection
form_recognizer_client = FormRecognizerClient(
    endpoint, AzureKeyCredential(subscription_key)
)

with open("images/business_card.png", "rb") as business_card_stream:
    poller = form_recognizer_client.begin_recognize_business_cards(business_card_stream)
    business_cards = poller.result()

    for idx, business_card in enumerate(business_cards):
        print("--------Analyzing business card #{}--------".format(idx + 1))
        contact_names = business_card.fields.get("ContactNames")
        if contact_names:
            for contact_name in contact_names.value:
                print(
                    "Contact First Name: {} has confidence: {}".format(
                        contact_name.value["FirstName"].value,
                        contact_name.value["FirstName"].confidence,
                    )
                )
                print(
                    "Contact Last Name: {} has confidence: {}".format(
                        contact_name.value["LastName"].value,
                        contact_name.value["LastName"].confidence,
                    )
                )
        company_names = business_card.fields.get("CompanyNames")
        if company_names:
            for company_name in company_names.value:
                print(
                    "Company Name: {} has confidence: {}".format(
                        company_name.value, company_name.confidence
                    )
                )
        departments = business_card.fields.get("Departments")
        if departments:
            for department in departments.value:
                print(
                    "Department: {} has confidence: {}".format(
                        department.value, department.confidence
                    )
                )
        job_titles = business_card.fields.get("JobTitles")
        if job_titles:
            for job_title in job_titles.value:
                print(
                    "Job Title: {} has confidence: {}".format(
                        job_title.value, job_title.confidence
                    )
                )
        emails = business_card.fields.get("Emails")
        if emails:
            for email in emails.value:
                print(
                    "Email: {} has confidence: {}".format(email.value, email.confidence)
                )
        websites = business_card.fields.get("Websites")
        if websites:
            for website in websites.value:
                print(
                    "Website: {} has confidence: {}".format(
                        website.value, website.confidence
                    )
                )
        addresses = business_card.fields.get("Addresses")
        if addresses:
            for address in addresses.value:
                print(
                    "Address: {} has confidence: {}".format(
                        address.value, address.confidence
                    )
                )
        mobile_phones = business_card.fields.get("MobilePhones")
        if mobile_phones:
            for phone in mobile_phones.value:
                print(
                    "Mobile phone number: {} has confidence: {}".format(
                        phone.value, phone.confidence
                    )
                )
        faxes = business_card.fields.get("Faxes")
        if faxes:
            for fax in faxes.value:
                print(
                    "Fax number: {} has confidence: {}".format(
                        fax.value, fax.confidence
                    )
                )
        work_phones = business_card.fields.get("WorkPhones")
        if work_phones:
            for work_phone in work_phones.value:
                print(
                    "Work phone number: {} has confidence: {}".format(
                        work_phone.value, work_phone.confidence
                    )
                )
        other_phones = business_card.fields.get("OtherPhones")
        if other_phones:
            for other_phone in other_phones.value:
                print(
                    "Other phone number: {} has confidence: {}".format(
                        other_phone.value, other_phone.confidence
                    )
                )
