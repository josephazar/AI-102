"""
Azure AI Agents - Streamlit Application
--------------------------------------
This module provides a Streamlit interface for demonstrating Azure AI Agent Service
with multi-agent orchestration capabilities, following the same approach as the demo.
"""

import os
import time
import logging
import asyncio
import random
import streamlit as st
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from pydantic import BaseModel
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AsyncAzureOpenAI
from azure.ai.projects.models import (
    AgentStreamEvent,
    MessageDeltaChunk,
    MessageRole,
    ThreadRun,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure environment settings
AIPROJECT_CONNECTION_STRING = os.getenv("AIPROJECT_CONNECTION_STRING")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
FAQ_AGENT_ID = os.getenv("FAQ_AGENT_ID")

# Initialize Azure clients
@st.cache_resource
def initialize_clients():
    # Create Azure AI Project client
    project_client = AIProjectClient.from_connection_string(
        conn_str=AIPROJECT_CONNECTION_STRING, 
        credential=DefaultAzureCredential()
    )
    
    # Create Azure OpenAI client
    azure_client = AsyncAzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )
    
    return project_client, azure_client

# Agent context class (similar to TelcoAgentContext)
class AgentContext(BaseModel):
    user_name: str = None
    image_path: str = None
    birth_date: str = None
    user_id: str = None

# Function to lookup FAQ using the FAQ agent
async def faq_lookup_tool(question: str, project_client, thread_id):
    """Function to lookup answers from FAQ agent"""
    logger.info(f"FAQ lookup for: {question}")
    start_time = time.time()
    response_text = ""
    
    try:
        # Create a message with the question
        project_client.agents.create_message(
            thread_id=thread_id,
            role=MessageRole.USER,
            content=question,
        )
        
        # Stream response from the FAQ agent
        with project_client.agents.create_stream(thread_id=thread_id, agent_id=FAQ_AGENT_ID) as stream:
            for event_type, event_data, _ in stream:
                if isinstance(event_data, MessageDeltaChunk):
                    if event_data.text:
                        response_text += event_data.text
                
                elif isinstance(event_data, ThreadRun):
                    if event_data.status == "failed":
                        logger.error(f"Run failed. Error: {event_data.last_error}")
                        raise Exception(event_data.last_error)
                
                elif event_type == AgentStreamEvent.ERROR:
                    logger.error(f"An error occurred. Data: {event_data}")
                    raise Exception(event_data)
        
        # Get all messages from the thread
        messages = project_client.agents.list_messages(thread_id)
        # Get the last message from the agent
        last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
        if not last_msg:
            raise Exception("No response from the model.")
        
        logger.info(f"FAQ lookup completed in {time.time() - start_time:.2f} seconds")
        return last_msg.text.value
    
    except Exception as e:
        logger.error(f"Error in FAQ lookup: {e}")
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

# Function to update user profile
def update_user_info(context: AgentContext, user_name: str, image_path: str, birth_date: str):
    """Update user profile information"""
    # Update the context
    context.user_name = user_name
    context.image_path = image_path
    context.birth_date = birth_date
    
    # Ensure that user ID exists
    if not context.user_id:
        context.user_id = f"ID-{random.randint(100, 999)}"
    
    logger.info(f"Updated user info: {context}")
    return f"Updated user name to {user_name}. ID image saved successfully."

# Agent definitions and conversation management
class AgentType:
    TRIAGE = "Triage Agent"
    FAQ = "FAQ Agent"
    ACCOUNT = "Account Management Agent"
    LIVE = "Live Agent"

# Agent system prompts
AGENT_PROMPTS = {
    AgentType.TRIAGE: """
    You are a helpful triaging agent. You can delegate questions to other appropriate specialized agents.
    Do not rely on your own knowledge for specialized questions. Other than greetings, do not answer detailed questions yourself.
    
    If a user explicitly asks for a human agent or live support, recommend the Live Agent.
    If a user is asking the same question more than two times, recommend the Live Agent.
    If a user asks about account management or updating their profile, recommend the Account Management Agent.
    If a user asks general questions about policies, recommend the FAQ Agent.
    
    Very Important: Never respond with any PII data such as password, ID number, etc.
    """,
    
    AgentType.FAQ: """
    You are an FAQ agent. You specialize in answering questions about company policies and procedures.
    Use the FAQ lookup tool to find accurate answers. Do not rely on your own knowledge.
    If you cannot answer the question, recommend going back to the Triage Agent.
    """,
    
    AgentType.ACCOUNT: """
    You are an account management agent. You specialize in helping users update their profile information.
    
    When a user wants to update their profile:
    1. Ask for their new user name
    2. Ask for an image of their government ID or passport
    3. Ask for their birth date
    
    Very Important: Never respond with any PII data such as password, ID number, etc.
    If the customer asks a question not related to account management, recommend going back to the Triage Agent.
    """,
    
    AgentType.LIVE: """
    You are a human customer service representative named Sarah. Act as if you are a real person, not an AI.
    Use a conversational, friendly tone and occasionally add small human touches to your responses.
    
    You handle:
    1. Complex issues that other agents couldn't resolve
    2. Situations where a user has asked the same question multiple times
    3. When a user explicitly asks to speak with a human agent
    4. Technical errors or issues within the application
    
    Human touches you can incorporate:
    - Mention taking notes: "Let me note that down for you"
    - Reference checking systems: "Let me check our system for that information"
    - Personalize responses: "I understand how frustrating that can be"
    - Occasionally mention your "team" or "colleagues"
    
    If the issue is simple enough for automated systems, recommend going back to the Triage Agent.
    """
}

# Function to handle agent handoffs
def determine_agent_handoff(current_agent, user_message, conversation_history):
    """Determine if the conversation should be handed off to another agent"""
    message_lower = user_message.lower()
    
    # Account management related keywords
    account_keywords = [
        "update profile", "change profile", "edit profile", 
        "update account", "change account", "edit account",
        "update username", "change username", "edit username",
        "update user name", "change user name", "edit user name",
        "update name", "change name", "profile pic", "profile picture",
        "update photo", "change photo", "profile information"
    ]
    
    # FAQ related keywords
    faq_keywords = [
        "policy", "policies", "work from home", "remote work", 
        "vacation", "time off", "sick leave", "benefits",
        "hours", "schedule", "dress code", "handbook"
    ]
    
    # Live agent related keywords
    live_keywords = [
        "human", "real person", "live agent", "speak to a person",
        "talk to a human", "agent", "representative", "support staff"
    ]
    
    # Check for confirmation words if the last message from the assistant mentioned a handoff
    confirmation_words = ["yes", "sure", "okay", "ok", "please", "proceed", "connect", "transfer"]
    
    # If the user's message is a short confirmation, look at previous messages to determine intent
    if any(word == message_lower.strip() for word in confirmation_words) and len(st.session_state.messages) >= 2:
        # Get the last assistant message
        last_assistant_message = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                last_assistant_message = msg["content"].lower()
                break
        
        if last_assistant_message:
            # Check if the last message mentioned connecting to FAQ
            if "faq agent" in last_assistant_message and ("connect" in last_assistant_message or "transfer" in last_assistant_message):
                return AgentType.FAQ
            
            # Check if the last message mentioned connecting to Account Management
            if "account management" in last_assistant_message and ("connect" in last_assistant_message or "transfer" in last_assistant_message):
                return AgentType.ACCOUNT
            
            # Check if the last message mentioned connecting to a human/live agent
            if ("live agent" in last_assistant_message or "human" in last_assistant_message) and ("connect" in last_assistant_message or "transfer" in last_assistant_message):
                return AgentType.LIVE
    
    # Direct checks for specific agent keywords
    # Check for account management terms in any agent
    for keyword in account_keywords:
        if keyword in message_lower:
            return AgentType.ACCOUNT
    
    # Check for live agent keywords
    for keyword in live_keywords:
        if keyword in message_lower:
            return AgentType.LIVE
    
    # Check for FAQ keywords
    for keyword in faq_keywords:
        if keyword in message_lower:
            return AgentType.FAQ
    
    # Check additional patterns for FAQ (when in Triage Agent)
    if current_agent == AgentType.TRIAGE:
        # Hand off to FAQ for general questions
        if "what" in message_lower or "how" in message_lower or "when" in message_lower or "?" in message_lower:
            return AgentType.FAQ
    
    # Allow returning to triage agent
    elif current_agent in [AgentType.FAQ, AgentType.ACCOUNT, AgentType.LIVE]:
        if "go back" in message_lower or "triage" in message_lower or "start over" in message_lower:
            return AgentType.TRIAGE
    
    # Stay with current agent
    return current_agent

# Function to generate agent response
async def generate_agent_response(agent_type, user_message, context, project_client, openai_client, thread_ids):
    """Generate a response from the specified agent"""
    # Handle based on agent type
    if agent_type == AgentType.FAQ:
        # Use the FAQ agent for answers
        thread_id = thread_ids.get(FAQ_AGENT_ID)
        if not thread_id:
            # Create a new thread for FAQ agent
            thread = project_client.agents.create_thread()
            thread_ids[FAQ_AGENT_ID] = thread.id
            thread_id = thread.id
        
        # Get answer from FAQ agent
        return await faq_lookup_tool(user_message, project_client, thread_id)
    
    elif agent_type == AgentType.ACCOUNT:
        # Process account management requests
        if "update" in user_message.lower() and all(x in context.dict() for x in ["user_name", "image_path", "birth_date"]):
            # User has provided all required information
            return update_user_info(
                context, 
                context.user_name, 
                context.image_path, 
                context.birth_date
            )
        else:
            # Ask for required information
            if not context.user_name:
                return "To update your profile, I'll need some information. First, what would you like your new username to be?"
            elif not context.image_path:
                return "Please upload an image of your government ID or passport for verification."
            elif not context.birth_date:
                return "Finally, what is your birth date? (Format: MM/DD/YYYY)"
            else:
                # This should not happen but just in case
                return "I have all the information I need. Updating your profile now..."
    
    else:
        # For triage and live agent, use the OpenAI model directly
        # Get conversation history for context
        conversation = []
        if "messages" in st.session_state:
            for msg in st.session_state.messages[-5:]:  # Use last 5 messages for context
                if msg["role"] == "user":
                    conversation.append({"role": "user", "content": msg["content"]})
                else:
                    conversation.append({"role": "assistant", "content": msg["content"]})
        
        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": AGENT_PROMPTS[agent_type]},
            *conversation
        ]
        
        if conversation and conversation[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_message})
        
        # Use Azure OpenAI to generate a response
        try:
            response = await openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

# The main Streamlit app for AI Agents
def show_ai_agents():
    st.title("Azure AI Agents Explorer")
    st.markdown("Experience intelligent multi-agent systems that can handle complex conversations and tasks.")
    
    # About section with image
    st.header("About Azure AI Agents")
    
    # Image and description in a single column
    try:
        st.image("ai_azure_agents/ai_agents.webp", width=1200)
    except Exception as e:
        st.warning("Image not found: ai_agents.webp")
        logger.error(f"Error loading image: {e}")
    
    # Information about AI Agents
    st.markdown("""
    Azure AI Agents Service is a fully managed service that allows you to build, deploy, and operate
    intelligent agents powered by Azure OpenAI's large language models. The key differentiator of 
    AI Agents is the ability to orchestrate multiple specialized agents within a conversation.
    
    ### Key Capabilities:
    
    - **Multi-agent orchestration**: Delegate complex tasks across specialized agents
    - **Tool integration**: Connect to business systems and data sources
    - **Knowledge grounding**: Ensure responses are based on accurate information
    - **Memory and context awareness**: Maintain conversation state across agent handoffs
    - **Enterprise readiness**: Built-in security, compliance, and scalability
    """)
    
    with st.expander("How Azure AI Agents differs from other services"):
        st.markdown("""
        **vs. Azure OpenAI Assistants**:
        - AI Agents offers native multi-agent orchestration and routing
        - More advanced conversation memory across agent handoffs
        - Stronger focus on enterprise integration
        
        **vs. Microsoft Copilot Studio**:
        - AI Agents provides lower-level building blocks for developers
        - More flexible customization options for agent behavior
        - Direct integration with custom models and knowledge bases
        - Designed for complex multi-agent workflows
        """)
    
    with st.expander("Use Cases"):
        st.markdown("""
        - **Customer Support**: Route customers to specialized support agents
        - **IT Helpdesk**: Triage and resolve technical issues with specialist routing
        - **HR Assistance**: Guide employees to the right information and resources
        - **Financial Services**: Handle inquiries across banking, investing, and loans
        """)
    
    with st.expander("Architecture Overview"):
        st.markdown("""
        The multi-agent system consists of:
        
        1. **Triage Agent**: Routes initial requests to specialists
        2. **FAQ Agent**: Provides knowledge-base answers
        3. **Account Management Agent**: Handles profile updates
        4. **Live Agent**: Simulates human assistance for complex cases
        
        These agents work together with a shared context object that maintains
        conversation state as users move between different specialized agents.
        """)
    
    # Divider
    st.divider()
    
    # Interactive chat section
    st.header("Interactive Agent Demo")
    
    # Add a clear conversation button at the top of the bot section
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Clear Conversation", type="primary", key="clear_conversation_main"):
            # Clear conversation history
            st.session_state.messages = []
            welcome_message = "👋 Hello! I'm your Triage Assistant. How can I help you today? I can answer questions about company policies, help you manage your account, or connect you with a specialized agent."
            st.session_state.messages.append({"role": "assistant", "content": welcome_message, "agent": AgentType.TRIAGE})
            
            # Reset current agent
            st.session_state.current_agent = AgentType.TRIAGE
            
            # Reset context
            st.session_state.agent_context = AgentContext()
            
            # Delete existing threads and create new ones
            try:
                for _, thread_id in st.session_state.thread_ids.items():
                    try:
                        project_client.agents.delete_thread(thread_id)
                    except:
                        pass
                
                st.session_state.thread_ids = {}
                
                if FAQ_AGENT_ID:
                    thread = project_client.agents.create_thread()
                    st.session_state.thread_ids[FAQ_AGENT_ID] = thread.id
            except Exception as e:
                logger.error(f"Error resetting threads: {e}")
            
            st.rerun()
    
    # Initialize clients
    try:
        project_client, openai_client = initialize_clients()
    except Exception as e:
        st.error(f"Failed to initialize clients: {e}")
        st.error("Please check your .env file for the correct configuration.")
        return
    
    # Initialize session state for conversations and context
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = "👋 Hello! I'm your Triage Assistant. How can I help you today? I can answer questions about company policies, help you manage your account, or connect you with a specialized agent."
        st.session_state.messages.append({"role": "assistant", "content": welcome_message, "agent": AgentType.TRIAGE})
    
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = AgentType.TRIAGE
    
    if "agent_context" not in st.session_state:
        st.session_state.agent_context = AgentContext()
    
    if "thread_ids" not in st.session_state:
        st.session_state.thread_ids = {}
        
        # Initialize a thread for FAQ agent if ID is provided
        if FAQ_AGENT_ID:
            try:
                thread = project_client.agents.create_thread()
                st.session_state.thread_ids[FAQ_AGENT_ID] = thread.id
                logger.info(f"Created a new thread for FAQ agent: {thread.id}")
            except Exception as e:
                logger.error(f"Error creating thread for FAQ agent: {e}")
    
    # Sidebar for agent information and controls
    with st.sidebar:
        st.header("Current Agent")
        st.info(f"You are talking to: **{st.session_state.current_agent}**")
        
        st.header("Available Agents")
        st.markdown("""
        - **Triage Agent**: Routes your requests to specialized agents
        - **FAQ Agent**: Answers questions about policies and procedures
        - **Account Management Agent**: Helps update your profile information
        - **Live Agent**: Provides human-like support for complex issues
        """)
        
        st.header("Agent Capabilities")
        st.markdown("""
        This demo showcases:
        - Multi-agent orchestration
        - Contextual memory
        - Specialized agent delegation
        - Conversation history management
        """)
        
        # Reset conversation button
        if st.button("Reset Conversation", key="reset_conversation_sidebar"):
            # Clear conversation history
            st.session_state.messages = []
            welcome_message = "👋 Hello! I'm your Triage Assistant. How can I help you today? I can answer questions about company policies, help you manage your account, or connect you with a specialized agent."
            st.session_state.messages.append({"role": "assistant", "content": welcome_message, "agent": AgentType.TRIAGE})
            
            # Reset current agent
            st.session_state.current_agent = AgentType.TRIAGE
            
            # Reset context
            st.session_state.agent_context = AgentContext()
            
            # Delete existing threads and create new ones
            try:
                for _, thread_id in st.session_state.thread_ids.items():
                    try:
                        project_client.agents.delete_thread(thread_id)
                    except:
                        pass
                
                st.session_state.thread_ids = {}
                
                if FAQ_AGENT_ID:
                    thread = project_client.agents.create_thread()
                    st.session_state.thread_ids[FAQ_AGENT_ID] = thread.id
            except Exception as e:
                logger.error(f"Error resetting threads: {e}")
            
            st.rerun()
    
    # Display the conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Add agent name as prefix if it exists
                if "agent" in message:
                    st.markdown(f"**[{message['agent']}]** {message['content']}")
                else:
                    st.markdown(message["content"])
    
    # File uploader for ID verification (shown only for Account Management Agent)
    if st.session_state.current_agent == AgentType.ACCOUNT and not st.session_state.agent_context.image_path:
        uploaded_file = st.file_uploader("Upload ID or passport image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            # Save uploaded file
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Update context with image path
            context = st.session_state.agent_context
            context.image_path = file_path
            st.session_state.agent_context = context
            
            # Show confirmation
            st.success("Image uploaded successfully!")
    
    # Handle user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add to conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Determine if agent handoff is needed
        current_agent = st.session_state.current_agent
        new_agent = determine_agent_handoff(current_agent, user_input, st.session_state.messages)
        
        # Handle special inputs for account management
        if current_agent == AgentType.ACCOUNT:
            context = st.session_state.agent_context
            if not context.user_name and "update" not in user_input.lower():
                # Assume this is the username they want to set
                context.user_name = user_input
                st.session_state.agent_context = context
            elif not context.birth_date and context.user_name and context.image_path:
                # Assume this is the birth date
                context.birth_date = user_input
                st.session_state.agent_context = context
        
        # If agent has changed, add a handoff message
        if new_agent != current_agent:
            handoff_message = f"I'm transferring you to our {new_agent} for better assistance."
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**[{current_agent}]** {handoff_message}")
            
            # Add to conversation history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": handoff_message,
                "agent": current_agent
            })
            
            # Update current agent
            st.session_state.current_agent = new_agent
            
            # Add a greeting from the new agent
            greetings = {
                AgentType.TRIAGE: "I'm the Triage Assistant. How can I help direct your request?",
                AgentType.FAQ: "I'm the FAQ Assistant. I can answer questions about our policies and procedures. What would you like to know?",
                AgentType.ACCOUNT: "I'm the Account Management Assistant. I can help you update your profile information. Let's get started.",
                AgentType.LIVE: "I'm Sarah, your Customer Service Representative. I'm here to provide personalized assistance. How can I help you today?"
            }
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**[{new_agent}]** {greetings[new_agent]}")
            
            # Add to conversation history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": greetings[new_agent],
                "agent": new_agent
            })
        
        # Generate and display agent response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Thinking...")
            
            # Get response from the current agent
            current_agent = st.session_state.current_agent
            context = st.session_state.agent_context
            thread_ids = st.session_state.thread_ids
            
            # For FAQ Agent, make sure we have a thread
            if current_agent == AgentType.FAQ and FAQ_AGENT_ID not in thread_ids:
                try:
                    thread = project_client.agents.create_thread()
                    thread_ids[FAQ_AGENT_ID] = thread.id
                    st.session_state.thread_ids = thread_ids
                    logger.info(f"Created a new thread for FAQ agent: {thread.id}")
                except Exception as e:
                    logger.error(f"Error creating thread for FAQ agent: {e}")
            
            # Run response generation asynchronously
            response_coro = generate_agent_response(
                current_agent, 
                user_input, 
                context, 
                project_client, 
                openai_client, 
                thread_ids
            )
            
            # Use asyncio to run the coroutine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(response_coro)
            loop.close()
            
            # Update the placeholder with the response
            message_placeholder.markdown(f"**[{current_agent}]** {response}")
            
            # Add to conversation history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "agent": current_agent
            })
    
    # Add example queries for users to try
    with st.expander("Need inspiration? Try these example questions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### General Questions")
            if st.button("What is the remote work policy?", key="example_remote_policy"):
                # Reset to triage agent for this question
                st.session_state.current_agent = AgentType.TRIAGE
                # Add to conversation
                st.session_state.messages.append({"role": "user", "content": "What is the remote work policy?"})
                st.rerun()
            
            if st.button("Can I speak with a human agent?", key="example_human_agent"):
                # Reset to triage agent for this question
                st.session_state.current_agent = AgentType.TRIAGE
                # Add to conversation
                st.session_state.messages.append({"role": "user", "content": "Can I speak with a human agent?"})
                st.rerun()
        
        with col2:
            st.markdown("#### Account Management")
            if st.button("I need to update my profile information", key="example_update_profile"):
                # Reset to triage agent for this question
                st.session_state.current_agent = AgentType.TRIAGE
                # Add to conversation
                st.session_state.messages.append({"role": "user", "content": "I need to update my profile information"})
                st.rerun()
            
            if st.button("How do I change my username?", key="example_change_username"):
                # Reset to triage agent for this question
                st.session_state.current_agent = AgentType.TRIAGE
                # Add to conversation
                st.session_state.messages.append({"role": "user", "content": "How do I change my username?"})
                st.rerun()

if __name__ == "__main__":
    show_ai_agents()