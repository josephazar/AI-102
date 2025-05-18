import os
import sys
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config_manager import load_config, save_config

# Load environment variables
load_dotenv()

def delete_content_violation_agent():
    """
    Delete the AI agent for content code violation detection
    and associated resources based on configuration in config file.
    """
    # Get project connection string from environment variables
    connection_string = os.getenv("AIPROJECT_CONNECTION_STRING")
    if not connection_string:
        print("Error: AIPROJECT_CONNECTION_STRING environment variable not set.")
        sys.exit(1)

    # Load configuration from config file
    config = load_config()
    
    agent_id = config.get("agent_id")
    vector_store_id = config.get("vector_store_id")
    file_id = config.get("file_id")
    
    if not (agent_id and vector_store_id and file_id):
        print("Error: Missing configuration in config file.")
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
    
    # Delete the vector store
    try:
        print(f"Deleting vector store with ID: {vector_store_id}")
        project_client.agents.delete_vector_store(vector_store_id)
        print("Vector store deleted successfully.")
    except Exception as e:
        print(f"Error deleting vector store: {str(e)}")
    
    # Delete the agent
    try:
        print(f"Deleting agent with ID: {agent_id}")
        project_client.agents.delete_agent(agent_id)
        print("Agent deleted successfully.")
    except Exception as e:
        print(f"Error deleting agent: {str(e)}")
    
    # Delete the file
    try:
        print(f"Deleting file with ID: {file_id}")
        project_client.agents.delete_file(file_id)
        print("File deleted successfully.")
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
    
    # Remove deleted resources from config
    for key in ["agent_id", "vector_store_id", "file_id"]:
        if key in config:
            del config[key]
    
    save_config(config)
    print("Configuration updated.")

if __name__ == "__main__":
    delete_content_violation_agent()