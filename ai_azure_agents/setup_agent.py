"""
Azure AI Agent Service - Agent Setup Script
-------------------------------------------
This script creates and configures an AI Agent on Azure AI Agent Service.
Run this script once to setup your agent and obtain the agent ID.
"""

import os
import time
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole

# Load environment variables
load_dotenv()

AIPROJECT_CONNECTION_STRING = os.getenv("AIPROJECT_CONNECTION_STRING")
if not AIPROJECT_CONNECTION_STRING:
    print("Error: AIPROJECT_CONNECTION_STRING environment variable not found.")
    print("Please add it to your .env file: AIPROJECT_CONNECTION_STRING=<your-foundry-project-region>.api.azureml.ms;<your-subscription-id>;<your-resource-group>;<your-foundry-project>")
    exit(1)

# Create Azure AI Project client
try:
    client = AIProjectClient.from_connection_string(
        conn_str=AIPROJECT_CONNECTION_STRING, 
        credential=DefaultAzureCredential()
    )
    print("✅ Successfully connected to Azure AI Project.")
except Exception as e:
    print(f"❌ Error connecting to Azure AI Project: {str(e)}")
    exit(1)

def create_faq_agent():
    """Create a new FAQ agent and return its ID"""
    
    print("Creating FAQ Agent...")
    
    # Define the agent
    agent_name = "HR FAQ Agent"
    agent_description = "An agent that answers questions about HR policies and procedures"
    
    # Define the system message
    system_message = (
        "You are an HR assistant who helps employees find information about company policies and procedures. "
        "Answer questions accurately and provide specific policy information when available. "
        "If you don't know the answer, politely explain that you don't have that information yet. "
        "Always be professional and courteous."
    )
    
    try:
        # Create the agent using keyword arguments directly
        agent = client.agents.create(
            display_name=agent_name,
            description=agent_description,
            system_message=system_message
        )
        agent_id = agent.id
        print(f"✅ Successfully created FAQ Agent with ID: {agent_id}")
        print("\nAdd this to your .env file:")
        print(f"FAQ_AGENT_ID={agent_id}")
        
        # Create a new thread for the agent
        thread = client.agents.create_thread()
        print(f"✅ Created a new thread with ID: {thread.id}")
        
        # Add sample questions and answers to train the agent
        sample_qa_pairs = [
            {
                "question": "What is the remote work policy?",
                "answer": "Our company offers a flexible remote work policy. Employees can work remotely up to 3 days per week with manager approval. For fully remote positions, please refer to your employment contract or speak with your HR representative."
            },
            {
                "question": "How many vacation days do I get?",
                "answer": "Full-time employees receive 20 days of paid vacation annually, accrued at 1.67 days per month. New employees can begin using vacation time after their 90-day probationary period. Please use the HR portal to request time off."
            },
            {
                "question": "What is the parental leave policy?",
                "answer": "Our company offers 12 weeks of paid parental leave for primary caregivers and 6 weeks for secondary caregivers. This benefit is available to all full-time employees who have been with the company for at least 12 months."
            }
        ]
        
        # Add messages to the thread to train the agent
        print("\nTraining the agent with sample Q&A pairs...")
        for pair in sample_qa_pairs:
            # Add the user question
            client.agents.create_message(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=pair["question"]
            )
            
            # Add the assistant answer
            client.agents.create_message(
                thread_id=thread.id,
                role=MessageRole.AGENT,
                content=pair["answer"]
            )
            
            print(f"✅ Added Q&A pair: {pair['question'][:30]}...")
            
        print("\nFAQ Agent setup complete! You can now use this agent in your Streamlit application.")
        return agent_id
    
    except Exception as e:
        print(f"❌ Error creating FAQ Agent: {str(e)}")
        return None

if __name__ == "__main__":
    create_faq_agent()