import time
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering.authoring import AuthoringClient
from dotenv import load_dotenv
import os
load_dotenv()

# Set your Language Service resource endpoint and API key
endpoint = os.getenv("LANGUAGE_SERVICE_ENDPOINT")
credential = AzureKeyCredential(os.getenv("LANGUAGE_SERVICE_KEY"))

# Initialize the AuthoringClient
authoring_client = AuthoringClient(endpoint, credential)

# Define project details
project_name = "question-answering"
project_details = {
    "description": "My first question answering project",
    "language": "en"
}

# Create the project
authoring_client.create_project(project_name, project_details)
print(f"Project '{project_name}' created successfully.")

# Add a data source to the project
source_url = "https://download.microsoft.com/download/7/B/1/7B10C82E-F520-4080-8516-5CF0D803EEE0/surface-book-user-guide-EN.pdf"
source = {
    "displayName": "Surface Book User Guide",
    "sourceUri": source_url,
    "sourceKind": "url"
}

poller = authoring_client.begin_update_sources(
    project_name=project_name,
    sources=[{"op": "add", "value": source}]
)
poller.result()
print("Data source added successfully.")

# Deploy the project
deployment_name = "production"
poller = authoring_client.begin_deploy_project(project_name, deployment_name)
while not poller.done():
    print("Deploying project...")
    time.sleep(5)
print(f"Project '{project_name}' deployed successfully.")
