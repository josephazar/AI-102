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

formUrl = (
    "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/master/sdk/formrecognizer/azure-ai"
    "-formrecognizer/tests/sample_forms/forms/Form_1.jpg"
)

poller = form_recognizer_client.begin_recognize_content_from_url(formUrl)
page = poller.result()

table = page[0].tables[0]  # page 1, table 1
print("Table found on page {}:".format(table.page_number))
for cell in table.cells:
    print("Cell text: {}".format(cell.text))
    print("Location: {}".format(cell.bounding_box))
    print("Confidence score: {}\n".format(cell.confidence))

