import yaml
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
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

# training resource client connection
form_training_client = FormTrainingClient(
    endpoint, AzureKeyCredential(subscription_key)
)
