from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient
from dotenv import load_dotenv
import os
load_dotenv()

# Set your Language Service resource endpoint and API key
endpoint = os.getenv("LANGUAGE_SERVICE_ENDPOINT")
credential = AzureKeyCredential(os.getenv("LANGUAGE_SERVICE_KEY"))


# Initialize the QuestionAnsweringClient
qa_client = QuestionAnsweringClient(endpoint, credential)

# Define the input text (document content)
input_text = """
The Surface Pro 4 is equipped with a lithium-ion battery that provides up to 9 hours of video playback.
Charging the device from 0% to 100% typically takes about 2 to 4 hours, depending on usage during charging.
It's recommended to use the official Surface charger to ensure optimal charging performance.
"""

# Define the question
question = "How long does it take to fully charge the Surface Pro 4?"

# Query the document
response = qa_client.get_answers_from_text(
    question=question,
    text_documents=[{"id": "1", "text": input_text}]
)

# Display the answers
for answer in response.answers:
    print(f"Confidence: {answer.confidence:.2f}")
    print(f"Answer: {answer.answer}")
    print()