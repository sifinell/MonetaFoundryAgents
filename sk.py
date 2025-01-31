from agents.agent_init import AgentInitializer
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from multiagent.agent_team import AgentTeam
from dotenv import load_dotenv
import os
from semantic_kernel.agents.open_ai.azure_assistant_agent import AzureAssistantAgent
from multiagent.azure_ai_agent_service import AzureAssistantAgent

import asyncio
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.kernel import Kernel

# Load environment variables from .env file
load_dotenv()

# Initialize Project client from Foundry
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

# Function to delete all agents from the project
def delete_agents():
    # List all agents from Project
    agents_list = project_client.agents.list_agents()

    # Check if agent already exists
    for agent in agents_list.data:
        project_client.agents.delete_agent(agent.id)

# Clean agents from Foundry
delete_agents()

async def main():

    # Class to initialize agents
    agent_initializer = AgentInitializer(project_client)

    # Agents creation
    agent_cio = agent_initializer.create_agent(agent_type="cio")
    agent_crm = agent_initializer.create_agent(agent_type="crm")
    agent_funds = agent_initializer.create_agent(agent_type="funds")
    #agent_news = agent_initializer.create_agent(agent_type="news")
    #agent_responder = agent_initializer.create_agent(agent_type="responder")

    kernel = Kernel()
    
    agent_cio_sk = await AzureAssistantAgent.retrieve_foundry(
        project_client=project_client,
        agent=agent_cio, 
        kernel=kernel)
        
    agent_crm_sk = await AzureAssistantAgent.retrieve_foundry(
        project_client=project_client,
        agent=agent_crm, 
        kernel=kernel)
    
    agent_funds_sk = await AzureAssistantAgent.retrieve_foundry(
        project_client=project_client,
        agent=agent_funds, 
        kernel=kernel)

    group_chat = AgentGroupChat(
        agents=[
            #agent_cio_sk,
            #agent_crm_sk,
            agent_funds_sk
        ]
    )

    while True:
        # Prompt the user for a question
        user_input = input("Enter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        await group_chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
        print(f"# User: '{user_input}'")

        async for content in group_chat.invoke():
            print(f"# Agent - {content.name or '*'}: '{content.content}'")

if __name__ == "__main__":
    asyncio.run(main())