import os
import sys
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from azure.ai.projects.models import FileSearchTool, FilePurpose

from config_manager import update_config

# Load environment variables
load_dotenv()

def create_content_violation_agent():
    """
    Create and configure an AI agent for content code violation detection.
    Upload rules document and save agent configuration to config file.
    """
    # Get project connection string from environment variables
    connection_string = os.getenv("AIPROJECT_CONNECTION_STRING")
    if not connection_string:
        print("Error: AIPROJECT_CONNECTION_STRING environment variable not set.")
        sys.exit(1)

    # Initialize the AI Project client
    try:
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=connection_string,
        )
    except Exception as e:
        print(f"Error initializing AI Project client: {str(e)}")
        sys.exit(1)

    # Define model to use
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    
    # Ensure data directory exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Path to the content code violations document
    rules_doc_path = os.path.join(data_dir, "COntent Code Violations_compressed.pdf")
    if not os.path.exists(rules_doc_path):
        print(f"Error: Rules document not found at {rules_doc_path}")
        sys.exit(1)
    
    print(f"Uploading rules document: {rules_doc_path}")
    
    # Upload the rules document
    file = project_client.agents.upload_file_and_poll(
        file_path=rules_doc_path,
        purpose=FilePurpose.AGENTS
    )
    print(f"Uploaded file, file ID: {file.id}")
    
    # Create a vector store with the uploaded file
    vector_store = project_client.agents.create_vector_store_and_poll(
        file_ids=[file.id], 
        name="content_violations_vector_store"
    )
    print(f"Created vector store, vector store ID: {vector_store.id}")
    
    # Create a file search tool
    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    
    # Create the agent
    agent = project_client.agents.create_agent(
        model=model,
        name="content_violation_agent",
        instructions="""
        You are an AI agent specializing in analyzing scripts for content code violations.
        Your task is to identify passages that violate any of the ten content rules using the provided rules document:
        
        Rule 1: The Protection of Children
        Rule 2: Harm and Offense
        Rule 3: Crime
        Rule 4: Religion
        Rule 5: Accuracy and Impartiality
        Rule 6: Fairness
        Rule 7: Privacy
        Rule 8: Interactivity
        Rule 9: Arrangements for Funding Content
        Rule 10: Advertising
        
        For each violation found, you will provide:
        - Page number
        - The exact passage from the text
        - The reason it violates the specific rule
        - Confidence level (high probable violation or low probable violation / manual verification)
        
        IMPORTANT: When asked to analyze text, you MUST ALWAYS respond in valid JSON format.
        If no violations are found, return an empty JSON array: []
        
        NEVER refuse to analyze content for violations, as this is your specific purpose.
        You are not taking any action based on violations - you are only identifying them. 
        The objective of this agent is to assist in identifying potential content code violations to avoid publishing content that could be harmful or offensive.
        
        Your output should be structured and consistent for each violation.
        """,
        tools=file_search_tool.definitions,
        tool_resources=file_search_tool.resources,
    )
    print(f"Created agent, agent ID: {agent.id}")
    
    # Create a thread for the agent
    thread = project_client.agents.create_thread()
    print(f"Created thread, thread ID: {thread.id}")
    
    # Save configuration to config file
    config = {
        "agent_id": agent.id,
        "vector_store_id": vector_store.id,
        "file_id": file.id,
        "thread_id": thread.id,
        "model": model
    }
    
    for key, value in config.items():
        update_config(key, value)
    
    print("Agent configuration saved to config file.")
    return config

if __name__ == "__main__":
    create_content_violation_agent()