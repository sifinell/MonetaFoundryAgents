from agents.agent_init import AgentInitializer
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from multiagent.agent_team import AgentTeam
from dotenv import load_dotenv
import os

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

# Function to test the execution of an agent
def test_agent(query, agent):

    # Create a thread
    thread = project_client.agents.create_thread()
    print(f"Created thread, thread ID: {thread.id}")
        
    # Create a message
    message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=query,
    )
    print(f"Created message, message ID: {message.id}")
            
    # Run the agent
    run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    print(f"Run finished with status: {run.status}")
        
    if run.status == "failed":
        # Check if you got "Rate limit is exceeded.", then you want to get more quota
        print(f"Run failed: {run.last_error}")

    # Get messages from the thread 
    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")
            
    assistant_message = ""
    for message in messages.data:
        if message["role"] == "assistant":
            assistant_message = message["content"][0]["text"]["value"]

    # Get the last message from the sender
    print(f"Assistant response: {assistant_message}")

# Function is provided in case dismantle_team is not executed. 
# If that is the case, Team lead would not be deleted and recreated every time assemble_team is executed
# Ideally TeamLeader should be retrieved if existing and passed to AgentTeam class so that it is not recreated on assemble_team
delete_agents()

# Class to initialize agents
agent_initializer = AgentInitializer(project_client)

# Agents creation
agent_cio = agent_initializer.create_agent(agent_type="cio")
agent_crm = agent_initializer.create_agent(agent_type="crm")
agent_funds = agent_initializer.create_agent(agent_type="funds")
agent_news = agent_initializer.create_agent(agent_type="news")
agent_responder = agent_initializer.create_agent(agent_type="responder")
agent_team_leader = agent_initializer.create_agent(agent_type="team_leader")

# Sample to test if single agents are working
test_agent("Provide me a summary of the portfolio's positions of my client id 123456", agent_crm)
test_agent("What are our Chief Investment Office (CIO) believes on the AI sector?", agent_cio)
test_agent("Can you give me an update on the UBS 100 Index Switzerland Equity Fund CHF and its latest performance?", agent_funds)

# Agents orchestration in Foundry
# Ideally to be replaced or with SK or enhanced by Vanilla, O1 Planner
with project_client:

    # Create a team of agents
    agent_team = AgentTeam("banking_team", project_client=project_client)

    # Add agents to the team, add_existing_agent was custom created in AgentTeam class to avoid the recreation of new Agents in Foundry 
    agent_team.add_existing_agent(agent_cio)
    agent_team.add_existing_agent(agent_crm)
    agent_team.add_existing_agent(agent_funds)
    agent_team.add_existing_agent(agent_news)
    agent_team.add_existing_agent(agent_responder)

    # Assemble the team, if no TeamLead is available it will be created
    agent_team.assemble_team()
    print("A team of specialized Banking agents is available for requests.")
    
    while True:
        user_input = input("Input (type 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        agent_team.process_request(request=user_input)

    # Dismantle the team, customized to avoid the deletion of Banking agents but to delete only TeamLead
    agent_team.dismantle_team()