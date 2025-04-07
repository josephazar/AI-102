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

receiptUrl = (
    "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/master/sdk/formrecognizer/azure-ai"
    "-formrecognizer/tests/sample_forms/receipt/contoso-receipt.png"
)
poller = form_recognizer_client.begin_recognize_receipts_from_url(receiptUrl)
result = poller.result()

for receipt in result:
    for name, field in receipt.fields.items():
        if name == "Items":
            print()
            print("Receipt Items:")
            for idx, items in enumerate(field.value):
                print()
                print("...Item #{}".format(idx + 1))
                for item_name, item in items.value.items():
                    print(
                        "......{}: {} has confidence {}".format(
                            item_name, item.value, item.confidence
                        )
                    )
        else:
            print()
            print(
                "{}: {} has confidence {}".format(name, field.value, field.confidence)
            )

