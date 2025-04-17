import os
import asyncio
from azure.ai.projects import AIProjectClient

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
import dotenv
from azure.ai.projects.models import ConnectionType
from azure.ai.projects.aio.operations import AgentsOperations

dotenv.load_dotenv()



project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"],
)
 
connections = project_client.connections.list()
for connection in connections:
    print(connection)



print("Default connection:")
connection = project_client.connections.get_default(
    connection_type=ConnectionType.AZURE_OPEN_AI,
    include_credentials=True,  # Optional. Defaults to "False".
)
print(connection)

print("List all agents:")
async def list_agents():
    agents = project_client.agents.list_agents()  # No need for await
    for agent in agents.data:  # Access the actual list of agents
        print(agent.id) 
        print("Deleting agent")
        try:
            project_client.agents.delete_agent(agent.id)
        except Exception as e:
            continue


asyncio.run(list_agents())